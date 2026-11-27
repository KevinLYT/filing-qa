"""Export one presentation figure from a scored run."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


SYSTEM_LABELS = {"B0": "B0   LLM only", "B1": "B1   Naive RAG", "B2": "B2   Single agent",
                 "Ours": "Ours   Three roles", "NoVerifier": "Ablation   No verifier"}
OUTCOMES = [("Correct", "#147D69", "white"), ("Incorrect", "#C35919", "white"),
            ("Refusal", "#B8C4D2", "#182633"), ("Run issue", "#59417C", "white")]
ISSUES = ("api_error", "parse_error", "input_error", "missing_prediction")


def render(run, output):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summaries = json.loads((run / "scores.json").read_text(encoding="utf-8"))["summary"]
    systems = manifest["systems"]
    n = len(manifest["question_ids"])
    if n == 0 or not systems or set(systems) != set(summaries):
        raise ValueError("The run needs scored results for all planned systems.")
    fig, (outcomes, latency) = plt.subplots(1, 2, sharey=True, figsize=(14.2, 7.5),
                                           gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=0.208, right=0.958, bottom=0.14, top=0.72, wspace=0.16)
    fig.text(0.055, 0.927, "Financial filing QA", fontsize=30, weight="bold", color="#152637")
    scope = {"dev": "development", "demo": "demo", "held-out": "held-out"}.get(manifest["scope"], manifest["scope"])
    model = manifest["model"]
    if model == "qwen2.5-3b-instruct-q4_k_m.gguf":
        model = "Qwen2.5-3B-Instruct (Q4_K_M)"
    noun = "question" if n == 1 else "questions"
    fig.text(0.055, 0.868, f"{n} {scope} {noun} per system   ·   {model}", fontsize=15, color="#41566B")
    legend = [Patch(facecolor=c, label=label) for label, c, _ in OUTCOMES]
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(0.049, 0.802), ncol=4,
               frameon=False, fontsize=13, handlelength=1.05, handleheight=0.85, columnspacing=1.65)
    outcomes.set_title("Questions by outcome", loc="left", fontsize=17, weight="bold", pad=14)
    latency.set_title("Mean time per question", loc="left", fontsize=17, weight="bold", pad=14)
    durations = [summaries[s]["mean_elapsed_seconds"] for s in systems]
    longest = max([v for v in durations if v is not None] + [1])
    for y, system in enumerate(systems):
        s = summaries[system]
        # Keep execution problems and refusals in the denominator, as in the scorer.
        counts = (s["correct"], s["incorrect"], s["refusal"], sum(s[k] for k in ISSUES))
        if s["n"] != n or sum(counts) != n:
            raise ValueError(f"Outcome counts disagree with the manifest for {system}.")
        left = 0
        for count, (_, color, ink) in zip(counts, OUTCOMES):
            if count:
                outcomes.barh(y, count, left=left, color=color, height=0.60, edgecolor="white", linewidth=0.8)
                outcomes.text(left + count / 2, y, str(count), ha="center", va="center",
                              fontsize=13, weight="bold", color=ink)
            left += count
        seconds = durations[y]
        if seconds is None:
            latency.text(0, y, "No timing recorded", va="center", fontsize=12)
        else:
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"Invalid recorded latency for {system}.")
            latency.barh(y, seconds, color="#254765", height=0.60)
            latency.text(seconds + longest * 0.022, y, f"{seconds:.2f} s", va="center", fontsize=13, color="#162E43")
    outcomes.set_yticks(range(len(systems)), labels=[SYSTEM_LABELS.get(s, s) for s in systems], fontsize=14)
    outcomes.set_ylim(len(systems) - 0.45, -0.58)
    outcomes.set_xlim(0, n)
    outcomes.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    latency.set_xlim(0, longest * 1.25)
    latency.xaxis.set_major_locator(MaxNLocator(nbins=5))
    for axis, label in [(outcomes, "Questions"), (latency, "Seconds")]:
        axis.set_xlabel(label, fontsize=13, labelpad=9, color="#3E5266")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#ADBBC6")
        axis.tick_params(axis="y", length=0, pad=10, colors="#182C3E")
        axis.tick_params(axis="x", colors="#526274", labelsize=12, length=3)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#DFE5EA", linewidth=0.7)
    if output.suffix.lower() != ".png":
        raise ValueError("Choose a PNG output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Directory containing the scored run")
    parser.add_argument("--out", required=True, type=Path, help="Output PNG file")
    args = parser.parse_args()
    render(args.run, args.out)
