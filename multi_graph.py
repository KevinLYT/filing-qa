"""Run three filing QA roles with a shared tool budget and at most one retrieval retry."""
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import multi_roles as roles
from run_agent import TOOL_LIMIT, model_turn, request_model

MODEL_LIMIT = TOOL_LIMIT + 1


class FilingState(TypedDict):
    attempt: int
    candidate: dict | None
    verified: bool
    feedback: str


class StopDialogue(Exception):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


class RoleContext:
    """Archive the actual model and MCP traffic for one question."""
    def __init__(self, trace, question, config, call_tool, complete):
        self.trace = trace
        self.question = question
        self.config = config
        self.call_tool = call_tool
        self.complete = complete
        self.step = None
        self.trace.update(graph_events=[], tool_call_count=0)

    def enter(self, role, attempt):
        self.role = role
        self.attempt = attempt
        self.node_tools = []
        self.step = None
        self.event("node")

    def event(self, kind, **fields):
        item = {"event": kind, "role": self.role, "attempt": self.attempt, **fields}
        self.trace["graph_events"].append(item)
        return item

    def excerpts(self):
        return "\n\n".join(f"[{c['chunk_id']} | PDF page {c['pdf_page']}]\n{c['text']}"
                            for c in self.trace["retrieved_chunks"])

    async def ask(self, role, prompt, user_content, schema):
        if role != self.role:
            raise ValueError("Model role does not match the current graph node")
        if self.trace["model_calls"] >= MODEL_LIMIT:
            raise StopDialogue("model_limit", "The per-question model-call limit was reached.")
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_content}]
        settings = {"model": self.config[1], "temperature": 0, "max_tokens": 800,
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": role.lower(), "strict": True, "schema": schema}}}
        self.event("model", step=len(self.trace["steps"]) + 1)
        try:
            self.step = await model_turn(self.trace, self.config, role, messages, settings, self.complete)
        finally:
            # Group citation checks with the verifier request that receives their results.
            self.trace["steps"][-1].update(attempt=self.attempt, tool_calls=self.node_tools, request_settings=settings)
        if self.step.get("provider_refusal") and not self.step["raw_response"]:
            raise StopDialogue("provider_refusal", str(self.step["provider_refusal"]))
        payload = json.loads(self.step["raw_response"])
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object for the role decision")
        return payload

    async def tool(self, name, arguments):
        if self.trace["tool_call_count"] >= TOOL_LIMIT:
            raise StopDialogue("tool_limit", "The tool budget was exhausted before a supported answer was ready.")
        self.trace["tool_call_count"] += 1
        self.trace["failure_stage"] = "mcp_tool"
        event = self.event("tool", name=name, arguments=arguments)
        record = {"name": name, "arguments": arguments}
        self.node_tools.append(record)
        result = await self.call_tool(name, arguments)
        event["result"] = result
        record["result"] = result
        if name == "search_filings":
            known = {hit["chunk_id"] for hit in self.trace["retrieved_chunks"]}
            for hit in result.get("chunks", []):
                if hit["chunk_id"] not in known:
                    self.trace["retrieved_chunks"].append(hit)
                    known.add(hit["chunk_id"])
        self.trace["failure_stage"] = "model_response"
        return result


def make_graph(ctx, with_verifier):
    async def retriever(state):
        attempt = state["attempt"] + 1
        ctx.enter("Retriever", attempt)
        ctx.feedback = state["feedback"]
        await roles.retrieve(ctx, ctx.feedback)
        return {"attempt": attempt, "candidate": None, "verified": False}

    async def calculator(state):
        ctx.enter("Calculator", state["attempt"])
        candidate, reason = await roles.calculate_candidate(ctx)
        return {"candidate": candidate, "feedback": reason}

    async def verifier(state):
        ctx.enter("Verifier", state["attempt"])
        passed, reason = await roles.verify_candidate(ctx, state["candidate"])
        return {"verified": passed, "feedback": reason}

    def decide(state):
        candidate = state["candidate"]
        if candidate is not None and with_verifier and ctx.role == "Calculator":
            return "Verifier"
        accepted = candidate is not None and (state["verified"] or not with_verifier)
        if accepted:
            decision = "accept"
        elif with_verifier and state["attempt"] < 2 and ctx.trace["tool_call_count"] < TOOL_LIMIT:
            decision = "retry"
        else:
            decision = "refuse"
        ctx.event("route", decision=decision, reason=state["feedback"])
        return "Retriever" if decision == "retry" else END

    graph = StateGraph(FilingState)
    graph.add_node("Retriever", retriever)
    graph.add_node("Calculator", calculator)
    graph.add_edge(START, "Retriever")
    graph.add_edge("Retriever", "Calculator")
    # Removing the verifier also removes its feedback edge; both upstream roles are identical.
    if with_verifier:
        graph.add_node("Verifier", verifier)
        graph.add_conditional_edges("Calculator", decide, ["Verifier", "Retriever", END])
        graph.add_conditional_edges("Verifier", decide, ["Retriever", END])
    else:
        graph.add_conditional_edges("Calculator", decide, [END])
    return graph.compile()


async def run_multi_dialogue(trace, question, config, call_tool, complete=request_model, verify=True):
    ctx = RoleContext(trace, question, config, call_tool, complete)
    start = {"attempt": 0, "candidate": None, "verified": False, "feedback": ""}
    try:
        state = await make_graph(ctx, verify).ainvoke(start, {"recursion_limit": 10})
        if state["candidate"] is not None and (state["verified"] or not verify):
            trace.update(state["candidate"], end_reason="verified" if verify else "verifier_removed")
            return
        reason = "verification_failed" if verify else "no_candidate"
        explanation = state["feedback"] or "The available evidence was insufficient."
    except StopDialogue as stop:
        reason, explanation = stop.reason, str(stop)
        ctx.event("route", decision="refuse", reason=explanation)
    trace.update(status="refusal", value=None, unit=None, citations=[], explanation=explanation, end_reason=reason)
