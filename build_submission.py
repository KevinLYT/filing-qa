"""Build and verify the offline class submission with the Python standard library."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PROJECT_FILES = """
.env.example .gitattributes .gitignore README.md README.zh-CN.md build_submission.py
requirements.txt start_qwen.ps1 analyze_errors.py b0.py b1.py check_dev_retrieval.py
comparison_report.py evaluate.py filing_mcp.py filing_numbers.py filing_tools.py hybrid_search.py
multi_graph.py multi_roles.py plot_results.py prepare_data.py replay.py retrieve_b1.py run_agent.py
run_baselines.py run_comparison.py score_b0.py setup_retrieval.py
data/questions.jsonl data/splits.json data/raw/source_tree.json
data/raw/financebench_open_source.jsonl data/raw/financebench_document_information.jsonl
data/evaluation/dev_operands.json data/evaluation/financebench_id_03031.json
data/evaluation/gold.jsonl data/evaluation/held_out_operands.json data/index/coverage.json
results/hybrid-dev-retrieval.json
""".split()
FINAL_FILES = """
FINAL_REPORT.pdf presentation/Financial_Filing_QA.pptx
presentation/Financial_Filing_QA.pdf
classify_failures.py results/held-out-final/failure_classifications.csv
""".split()
TESTS = """
test_agent_evaluation test_analyze_errors test_comparison_runner test_evaluate test_filing_tools
test_held_out_evaluation test_hybrid_search test_mcp_transport test_multi_evaluation test_multi_graph
test_multi_roles test_run_agent test_run_baselines
""".split()
RUNS = ("phase1-final", "hybrid-dev", "held-out-final")
RUN_ARTIFACTS = {
    "ai_review_suggestions.json", "dependencies.txt", "error_analysis.csv", "evaluation_manifest.json",
    "human_reviews.json", "manifest.json", "outcomes-and-time.png", "replay.txt",
    "replay-costco.txt", "replay-netflix.txt", "scores.json",
}
SYSTEMS = {"B0", "B1", "B2", "Ours", "NoVerifier"}
MANIFEST = "PACKAGE_MANIFEST.json"
SECRET = re.compile(rb"sk-[A-Za-z0-9_-]{25,}|Bearer [A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def archive_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or "\\" in name or ":" in name or ".." in path.parts or name != path.as_posix():
        raise ValueError(f"Invalid relative archive path: {name}")
    return path


def selected_run_file(path, ids):
    if len(path.parts) == 1:
        return path.name in RUN_ARTIFACTS
    return len(path.parts) == 2 and path.parts[0] in SYSTEMS and path.suffix == ".json" and path.stem in ids


def collect(project, extras, final_files):
    # An explicit release list keeps practice runs and machine-local files private.
    names = set(PROJECT_FILES + final_files)
    names.update(f"tests/{name}.py" for name in TESTS)
    questions = [json.loads(line) for line in (project / "data/questions.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    ids = {row["question_id"] for row in questions}
    names.update("data/index/" + row["doc_name"] + ".jsonl" for row in questions)
    for run in RUNS:
        folder = project / "results" / run
        for file in folder.rglob("*"):
            if file.is_file() and selected_run_file(file.relative_to(folder), ids):
                names.add(file.relative_to(project).as_posix())
    for name in extras:
        path = archive_path(name)
        if any(part.startswith((".", "review-practice-")) or part in ("models", "tools") for part in path.parts):
            raise ValueError(f"Unsupported optional file: {name}")
        if path.suffix not in (".py", ".md", ".json", ".csv", ".pdf", ".pptx", ".txt"):
            raise ValueError(f"Unsupported optional extension: {name}")
        names.add(name)
    payload = {}
    for name in sorted(names):
        archive_path(name)
        path = project / name
        if not path.resolve().is_relative_to(project) or not path.is_file():
            raise ValueError(f"Missing release file or external path: {name}")
        content = path.read_bytes()
        if SECRET.search(content):
            raise ValueError(f"Possible credential in release candidate: {name}")
        payload[name] = content
    for run, total in [("held-out-final", 26), ("hybrid-dev", 7)]:
        plan = json.loads(payload[f"results/{run}/manifest.json"])
        if set(plan["systems"]) != SYSTEMS or len(plan["question_ids"]) != total:
            raise ValueError(f"Unexpected frozen run plan: {run}")
        for system in SYSTEMS:
            for qid in plan["question_ids"]:
                member = f"results/{run}/{system}/{qid}.json"
                if member not in payload:
                    raise ValueError(f"Missing prediction: {member}")
    return payload


def verify(archive):
    with ZipFile(archive, "r") as zipped:
        members = zipped.namelist()
        manifest = json.loads(zipped.read(MANIFEST))
        files = manifest["files"]
        if len(members) != len(set(members)) or set(members) != set(files) | {MANIFEST}:
            raise ValueError("The ZIP entry list differs from the stored manifest.")
        for name, item in files.items():
            archive_path(name)
            raw = zipped.read(name)
            if len(raw) != item["bytes"] or digest(raw) != item["sha256"]:
                raise ValueError(f"Checksum or length mismatch: {name}")
    return len(files)


def build(project, out, extras, final_files):
    data = collect(project, extras, final_files)
    manifest = {"algorithm":"SHA-256", "financebench_revision":"cc39aeb4afdf33909ee1412188bf89035950c2eb",
                "optional_files":sorted(extras), "files":{n:{"bytes":len(data[n]), "sha256":digest(data[n])} for n in sorted(data)}}
    data[MANIFEST] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as folder:
        candidate = Path(folder) / "submission.zip"
        with ZipFile(candidate, "w", compression=ZIP_DEFLATED) as zipped:
            for name, raw in sorted(data.items()):
                info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0o600 << 16
                zipped.writestr(info, raw)
        count = verify(candidate)
        checksum = digest(candidate.read_bytes())
        checksum_path = out.with_name(out.name + ".sha256")
        checksum_text = f"{checksum}  {out.name}\n"
        if checksum_path.exists() and checksum_path.read_text(encoding="ascii") != checksum_text:
            raise FileExistsError("An existing checksum differs. Select a new output ZIP filename.")
        if out.exists():
            if digest(out.read_bytes()) != checksum:
                raise FileExistsError("This output already exists with different bytes. Select a new ZIP filename.")
        else:
            # Atomic rename on Windows refuses to replace an output that appeared concurrently.
            if os.name == "nt":
                candidate.rename(out)
            else:
                os.link(candidate, out)
        if not checksum_path.exists():
            with checksum_path.open("x", encoding="ascii", newline="\n") as stream:
                stream.write(checksum_text)
    print(f"Verified {count} payload files. ZIP bytes: {out.stat().st_size}. SHA-256: {checksum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Destination ZIP. New versions should use a new filename.")
    mode.add_argument("--verify", type=Path, help="Check each archive member against its embedded checksum")
    parser.add_argument("--project", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--include", action="append", default=[], help="Add a named, project-relative deliverable")
    args = parser.parse_args()
    if args.verify:
        print(f"Verified {verify(args.verify)} payload files.")
    else:
        if args.output.suffix.lower() != ".zip":
            parser.error("--output must have a .zip extension")
        build(args.project.resolve(), args.output.resolve(), args.include, FINAL_FILES)
