# Multi-Agent RAG for Numerically Grounded Financial Filing QA

[Chinese README](README.zh-CN.md) · [Research report](FINAL_REPORT.pdf) · [Presentation with speaker notes](presentation/Financial_Filing_QA.pptx) · [Slide PDF](presentation/Financial_Filing_QA.pdf)

New York University · CSCI-GA 3033-091, Introduction to LLM based Generative AI Systems · Fall 2026  
Yujun (Bruce) Chen and Yitan (Kevin) Li

**Does dividing retrieval, calculation and verification among three agents improve numerical answers to questions about annual filings?** We compared five systems using the same pretrained Qwen model. On 26 held-out FinanceBench questions, the single agent and the three-role workflow each scored 0/26. The single agent submitted 15 wrong answers, while the three-role workflow refused all 26. A one-call retrieval baseline scored 4/26. These results apply to the model, question selection and tool budgets described below.

## Replay a saved experiment

Open the folder containing `replay.py` in File Explorer, type `powershell` in the address bar and press Enter. Select the project interpreter, an installed Python 3.11 or the Windows Python launcher. These commands read saved predictions using only the standard library. They work offline without a model server, GPU or API key.

```powershell
$py = ".\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" }
if (!(Test-Path $py)) { $py = py -3.11 -c "import sys; print(sys.executable)" }
& $py -X utf8 replay.py --run results/phase1-final --id financebench_id_03031
& $py -X utf8 replay.py --run results/held-out-final --id financebench_id_03282
& $py -X utf8 comparison_report.py --run results/held-out-final
```

The first replay shows the development question on Lockheed Martin's FY2021 net working capital. B0 saw only the question. B1 retrieved the balance sheet as its top result, `LOCKHEEDMARTIN_2021_10K:p68:c1`. The 2021 column gives current assets of 19,815 and current liabilities of 13,997, both in USD millions. Their difference is 5,818. B1 returned **5,838**, within the allowed error of 58.18, but its two invented chunk IDs failed citation validation. Add `--full` to see the exact inputs and responses. [Original filing, PDF page 68](https://github.com/patronus-ai/financebench/blob/cc39aeb4afdf33909ee1412188bf89035950c2eb/pdfs/LOCKHEEDMARTIN_2021_10K.pdf#page=68)

The second replay is the Netflix case in the presentation. Page 45 reports current liabilities of 5,466,312 **thousand** dollars, equivalent to 5,466.312 million. Compare the three-role workflow's refusal with the ablation's submitted value of 8.796108 million. A [plain-text transcript](results/held-out-final/replay-netflix.txt) also works without Python.

## Experimental design

| System | Information and actions available |
|---|---|
| B0 · question only | One request containing the question and common answer instructions. |
| B1 · one-pass retrieval | The same request plus four BM25-ranked passages from the assigned filing. |
| B2 · single agent | One agent selects searches, calculator operations and citation checks over multiple turns. |
| Ours · three roles | A LangGraph Retriever selects a query, a Calculator selects operands for Python, and a Verifier checks evidence. A failed attempt can trigger one retrieval retry within budget. |
| NoVerifier · ablation | Uses the same Retriever and Calculator and submits a valid candidate immediately. Removes verification and its retry. |

B2, Ours and NoVerifier use the same MCP tools. Search combines BM25 and a dense MiniLM encoder, fuses ranks, then uses a cross-encoder to select two passages from eight candidates. Each system uses Qwen2.5-3B-Instruct Q4_K_M, temperature 0 and up to 800 generated tokens per call. Each agent workflow shares a limit of six tool calls and seven model calls per question. Role prompts, validation and control flow differ between B2 and Ours, so the experiment measures the effect of the complete workflows.

The pinned FinanceBench release contains 150 public questions. Of the 112 linked to 10-Ks, 33 had a single scorable percentage or an explicitly scaled dollar answer. Seven development questions cover six filings. The 26 held-out questions cover 24 other filings. The released question identifies the filing, and its metadata supplies company and year. Retrieval searches the **entire assigned filing**. Gold answers and evidence pages are reserved for evaluation. The text index contains 7,432 chunks from 4,850 PDF pages, with no text extracted from 49 pages. Physical PDF pages start at 1 and program page indices start at 0. [FinanceBench release](https://github.com/patronus-ai/financebench/tree/cc39aeb4afdf33909ee1412188bf89035950c2eb) · [Selection and split](data/splits.json) · [Index coverage](data/index/coverage.json)

## Held-out results

| System | Correct | Incorrect | Refusal | Parse failure | Mean seconds/question | Model calls |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 22 | 3 | 1 | 3.13 | 26 |
| B1 | 4 | 10 | 8 | 4 | 3.42 | 26 |
| B2 | 0 | 15 | 5 | 6 | 10.54 | 157 |
| Ours | 0 | 0 | 26 | 0 | 7.92 | 125 |
| NoVerifier | 0 | 12 | 14 | 0 | 3.49 | 52 |

All 130 attempts were recorded. All four numerically correct answers failed the automatic citation audit. There were zero API failures, input failures or missing predictions. Local provider charges were $0, excluding electricity and hardware. The 41 submitted answers from retrieval-based systems still await signed human judgments on year, line item and unit. An AI-assisted audit is recorded separately. [Scores](results/held-out-final/scores.json) · [Failure classifications](results/held-out-final/failure_classifications.csv) · [Human review fields](results/held-out-final/human_reviews.json)

## Run a fresh experiment on Windows

**Prepare the environment and retrieval models.** The experiment ran with Python 3.11.9 on Windows x64 using an RTX 4090 Laptop GPU for generation. In the same PowerShell session, with `$py` selected above:

```powershell
if (!(Test-Path .\.venv\Scripts\python.exe)) { & $py -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r results/held-out-final/dependencies.txt
& $py -m unittest discover -s tests -v
& $py setup_retrieval.py
& $py hybrid_search.py build --scope eligible
```

The release includes searchable text for the 30 eligible filings. To rebuild it from the original PDFs, run `& $py prepare_data.py all` **before** building the hybrid index. This fetches and verifies both official JSONL files and 84 source PDFs, applies the selection rule and re-extracts the eligible filings. To restore just the PDFs under `data/filings/`, use `& $py prepare_data.py fetch`. File names and upstream blob hashes are in the [source manifest](data/raw/source_tree.json).

**Start the local model in PowerShell A.** The first launch downloads and verifies the pinned Qwen weights and llama.cpp b7999 Windows x64 Vulkan runtime. Keep the process listening on `127.0.0.1:18091`. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/7dabda4d13d513e3e842b20f0d435c732f172cbe) · [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b7999)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_qwen.ps1
```

**Run one new development question in PowerShell B** from the repository root. Prediction calls the model. Evaluation scores the saved files independently.

```powershell
$py = ".\.venv\Scripts\python.exe"
$run = "results/demo-$(Get-Date -Format yyyyMMdd-HHmmssfff)"
& $py run_comparison.py --scope demo --retrieval hybrid --systems B0 B1 B2 Ours NoVerifier --out $run
& $py evaluate.py --run $run
& $py replay.py --run $run
```

**Repeat the full paired experiment** in PowerShell B. The held-out runner checks that the model, dependencies, prediction source and retrieval settings match the new development run. The original manifests contain hashes of the source bytes used at the time. Editing a comment changes the hash. Use the newly generated development manifest for a new held-out run.

```powershell
$py = ".\.venv\Scripts\python.exe"
$stamp = Get-Date -Format yyyyMMdd-HHmmssfff
$dev = "results/dev-$stamp"
$test = "results/held-out-$stamp"
& $py run_comparison.py --scope dev --retrieval hybrid --systems B0 B1 B2 Ours NoVerifier --out $dev
& $py evaluate.py --run $dev
& $py run_comparison.py --scope held-out --retrieval hybrid --systems B0 B1 B2 Ours NoVerifier --out $test --frozen-from "$dev/manifest.json"
& $py evaluate.py --run $test
& $py comparison_report.py --run $test
```

To continue an interrupted prediction run, repeat its command with the **same output path** and add `--resume`. For an offline export of the B0/B1 inputs, run `& $py run_baselines.py export --scope dev` after installing the dependencies. The export produces prompts only.

**Optional hosted model.** For a compatible Chat Completions endpoint with strict JSON Schema support, copy `.env.example` to `.env` and supply `OPENAI_API_KEY`, `QA_MODEL` and `QA_BASE_URL`. Add `--hosted` to the prediction command. Environment variables take precedence over the file. A cost estimate uses `QA_PRICE_MODEL` matching the model ID, a dated `QA_PRICE_SOURCE`, and USD per million tokens in `QA_INPUT_PRICE`, `QA_OUTPUT_PRICE` and, when cached tokens are billed, `QA_CACHED_INPUT_PRICE`. Missing usage or required rates produce `unknown`.

## Inspect and rescore predictions

- **Saved inputs and outputs.** Open `results/<run>/<system>/<question_id>.json` for messages, retrieved text, ranks, generated answers, model calls, tokens and latency. `manifest.json` records the planned IDs, model and protocol. `scores.json` contains every numeric and citation grade.
- **Numeric scoring.** Amounts are converted to USD and accepted when the absolute error is at most 1% of the absolute gold value. A zero amount requires exact zero. Percentage values accept an absolute error of at most 0.1 percentage points. `20 percent` and `0.20 fraction` represent the same rate. Changes expressed in `percentage_points` have their own unit.
- **Citation scoring.** Each source figure must appear in a cited passage actually sent to the model, with the correct document, physical page and chunk ID. Citations must cover a complete reference operand path. A calculated result may be absent from the filing. The denominator includes every scheduled question, including refusals and run failures.
- **Code entry points.** Follow a request through [prediction](run_comparison.py), [hybrid search](hybrid_search.py), [shared tools](filing_tools.py), the [three-role graph](multi_graph.py) and [scoring](evaluate.py).

To try a new scoring rule while preserving the archived scores, make a fresh copy of the run and evaluate it offline.

```powershell
$copy = "results/rescore-$(Get-Date -Format yyyyMMdd-HHmmssfff)"
Copy-Item -LiteralPath results/held-out-final -Destination $copy -Recurse
& $py -X utf8 evaluate.py --run $copy
```

## Package the project

From the repository root, build a ZIP with the two READMEs, source code, indexes, saved experiments, report and slides. The builder records a SHA-256 hash for each file and verifies the archive before saving it.

```powershell
& $py build_submission.py --output .\submission\filing-qa.zip
& $py build_submission.py --verify .\submission\filing-qa.zip
```
