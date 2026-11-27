# 面向财报数值问答的多智能体检索增强生成

[英文版](README.md) · [英文研究报告](FINAL_REPORT.pdf) · [含讲者备注的幻灯片](presentation/Financial_Filing_QA.pptx) · [幻灯片 PDF](presentation/Financial_Filing_QA.pdf)

纽约大学 · CSCI-GA 3033-091《Introduction to LLM based Generative AI Systems》· 2026 年秋  
团队成员：Yujun (Bruce) Chen、Yitan (Kevin) Li

**回答财报中的数值问题时，把找证据、计算、核验分给三个智能体，能提高准确率吗？** 我们用同一个预训练好的 Qwen 模型比较了五种系统。在 26 道留出题上，单智能体和三角色流程均为 0/26。单智能体交了 15 个错误答案，三角色流程拒答全部 26 题。只检索一次的基线答对了 4 题。这个结论限于本实验采用的模型、题目与调用预算。

## 重放一次已经完成的实验

用资源管理器打开含有 `replay.py` 的文件夹，在地址栏输入 `powershell`，按回车。下面前三行选择 Python 3.11 解释器，后三行分别重放两道题并输出总成绩。重放只读取本地记录，使用 Python 标准库即可，无需启动模型、连接网络或填写 API 密钥。

```powershell
$py = ".\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" }
if (!(Test-Path $py)) { $py = py -3.11 -c "import sys; print(sys.executable)" }
& $py -X utf8 replay.py --run results/phase1-final --id financebench_id_03031
& $py -X utf8 replay.py --run results/held-out-final --id financebench_id_03282
& $py -X utf8 comparison_report.py --run results/held-out-final
```

**第一题看懂输入与评分。** 洛克希德·马丁的开发题问 2021 财年净营运资本。B0 只看到题目。B1 检索整份年报，排名第一的片段是 `LOCKHEEDMARTIN_2021_10K:p68:c1`，即 PDF 第 68 页、程序索引 67 的资产负债表。看 2021 列，以百万美元计，流动资产 19,815 减流动负债 13,997 等于 **5,818**。B1 的预测是 **5,838**，相差 20，小于允许误差 58.18，数值判对。它写的两个片段 ID 均不存在于输入，引用判错。在重放命令末尾加 `--full` 可核对模型看到的每条消息及原始回复。[年报原页](https://github.com/patronus-ai/financebench/blob/cc39aeb4afdf33909ee1412188bf89035950c2eb/pdfs/LOCKHEEDMARTIN_2021_10K.pdf#page=68)

**第二题用于课堂演示。** Netflix 年报第 45 页写的是流动负债 5,466,312 **千美元**，换成百万美元为 5,466.312。观看重放时可以指出三角色流程在用尽预算时拒答，移除核验的版本则提交了错误结果 8.796108 百万美元。现场只有幻灯片软件时，也可以直接打开[英文纯文本重放记录](results/held-out-final/replay-netflix.txt)。

## 实验设计

| 系统 | 作答时可以看到什么、做什么 |
|---|---|
| B0：仅模型 | 一次调用，仅提供问题和共同的作答说明。 |
| B1：单次检索 | 在相同输入上增加四段原文，由程序在指定财报内用 BM25 排出。 |
| B2：单智能体 | 同一个智能体多轮决定何时检索、计算和检查引用。 |
| Ours：三角色 | LangGraph 依次调度检索、计算和核验三个角色。计算角色选取操作数，Python 执行运算。一次尝试失败后可在预算内重新检索一次。 |
| NoVerifier：消融 | 复用相同的检索与计算角色，遇到有效候选就交卷，移除核验与对应的重试。 |

三种智能体系统通过同一个 MCP 服务使用工具。检索结合 BM25 关键词匹配与 MiniLM 向量检索，融合两个排名，从八个候选中用交叉编码器选出两段。五套系统的回答模型均为 Qwen2.5-3B-Instruct Q4_K_M，温度为 0，每次最多生成 800 个 token。每个智能体系统在一道题内共用六次工具调用、七次模型调用的上限。B2 和三角色流程的提示、校验与执行顺序也有差异，故成绩衡量的是两套完整流程。

题目与年报来自 FinanceBench 官方固定版本。150 道公开题中，112 道对应 10-K，进一步筛出 33 道答案为单一百分数、或题中明确指定美元数量尺度的数值题。开发集 7 题涉及 6 份财报，留出集 26 题涉及另外 24 份。题目元数据指定文件，官方文档元数据给出公司和财年；检索覆盖**整份指定年报**。标准答案和标注证据页只用于评分。30 份财报共有 4,850 个 PDF 页面，得到 7,432 段，其中 49 页未提取到文字。PDF 物理页码从 1 开始，程序页索引从 0 开始。[FinanceBench 官方版本](https://github.com/patronus-ai/financebench/tree/cc39aeb4afdf33909ee1412188bf89035950c2eb) · [题目划分](data/splits.json) · [索引覆盖记录](data/index/coverage.json)

## 留出集成绩

| 系统 | 正确 | 错误 | 拒答 | 格式失败 | 平均秒数/题 | 模型总调用次数 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 22 | 3 | 1 | 3.13 | 26 |
| B1 | 4 | 10 | 8 | 4 | 3.42 | 26 |
| B2 | 0 | 15 | 5 | 6 | 10.54 | 157 |
| Ours | 0 | 0 | 26 | 0 | 7.92 | 125 |
| NoVerifier | 0 | 12 | 14 | 0 | 3.49 | 52 |

全部 130 次作答都有记录，API 失败、输入失败及缺失答卷均为 0。四个数值正确的答案均未通过自动引用检查。本地推理产生的供应商 API 费用为 0 美元，不含电费和设备成本。使用检索且实际交卷的 41 份答案仍待团队成员签署年份、科目与单位的人工判断；辅助审阅结果单独保存。[逐题评分](results/held-out-final/scores.json) · [逐题错误分类](results/held-out-final/failure_classifications.csv) · [人工复核字段](results/held-out-final/human_reviews.json)

## 在 Windows 上发起全新运行

**准备环境与检索模型。** 原实验使用 Windows x64、Python 3.11.9 和 RTX 4090 Laptop GPU。在上方已设置 `$py` 的 PowerShell 中，从项目根目录执行：

```powershell
if (!(Test-Path .\.venv\Scripts\python.exe)) { & $py -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r results/held-out-final/dependencies.txt
& $py -m unittest discover -s tests -v
& $py setup_retrieval.py
& $py hybrid_search.py build --scope eligible
```

仓库带有 30 份入选年报的文本索引。如果要从原始 PDF 完整重建，在生成混合索引前运行 `& $py prepare_data.py all`。这会下载并核验两个官方 JSONL 文件和 84 份原始 PDF，再筛题、解析入选财报。如只需把 PDF 恢复到 `data/filings/`，运行 `& $py prepare_data.py fetch`。原始文件名、版本及哈希见[来源清单](data/raw/source_tree.json)。

**在 PowerShell A 启动模型。** 首次启动会下载并核验指定的 Qwen 权重与 llama.cpp b7999 Windows x64 Vulkan 程序。保持窗口开启，模型服务在 `127.0.0.1:18091` 监听。[Qwen 权重版本](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/7dabda4d13d513e3e842b20f0d435c732f172cbe) · [llama.cpp 版本](https://github.com/ggml-org/llama.cpp/releases/tag/b7999)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_qwen.ps1
```

**在 PowerShell B 实时跑一道开发题。** 从项目根目录打开第二个窗口。预测命令发起模型调用；评分和重放读取已经写入硬盘的答卷。

```powershell
$py = ".\.venv\Scripts\python.exe"
$run = "results/demo-$(Get-Date -Format yyyyMMdd-HHmmssfff)"
& $py run_comparison.py --scope demo --retrieval hybrid --systems B0 B1 B2 Ours NoVerifier --out $run
& $py evaluate.py --run $run
& $py replay.py --run $run
```

**在 PowerShell B 重跑完整实验。** 先跑完开发集，再用同一模型、依赖、预测代码和检索设置跑留出集。原实验清单保存了当时源码文件的字节哈希，修改一行注释也会改变哈希。本次新运行应使用刚刚生成的开发集清单。

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

如预测中断，用**原来的输出路径**重新执行那条预测命令，末尾加 `--resume`。只想看 B0/B1 的输入，可在装好依赖后执行 `& $py run_baselines.py export --scope dev`，它只写提示，不会调用模型。

**可选的托管模型。** 服务需兼容 Chat Completions 并支持严格的 JSON Schema。将 `.env.example` 复制为 `.env`，填写 `OPENAI_API_KEY`、`QA_MODEL`、`QA_BASE_URL`，在预测命令末尾加 `--hosted`。环境变量优先于文件配置。估算费用需提供与模型 ID 一致的 `QA_PRICE_MODEL`、含日期的价格来源 `QA_PRICE_SOURCE`，以及每百万 token 的美元单价 `QA_INPUT_PRICE`、`QA_OUTPUT_PRICE`。有缓存输入用量时还需 `QA_CACHED_INPUT_PRICE`。缺少用量或必要价格时记录为 `unknown`。

## 阅读答卷与重新评分

- **看模型收到了什么。** 打开 `results/<批次>/<系统>/<题号>.json`，可看到消息、检索原文及排名、模型回答、调用次数、token 用量和耗时。批次目录下的 `manifest.json` 保存预定题目和实验配置，`scores.json` 保存数值和引用成绩。
- **判断数值。** 金额统一成美元后，允许绝对误差不超过标准答案绝对值的 1%。零美元要求精确零。百分数允许最多 0.1 个百分点误差。`20 percent` 与 `0.20 fraction` 均表示 20%；以百分点表示的变化使用单独单位 `percentage_points`。
- **判断引用。** 引用须指向模型实际收到的文档、PDF 页及片段，原文中必须有声称使用的数值，并覆盖一组能支持该答案的参考操作数。计算结果本身可以不在年报原文中。拒答、运行失败和缺失记录均保留在整体准确率分母中。
- **读代码。** 按实际作答顺序阅读[统一预测入口](run_comparison.py)、[混合检索](hybrid_search.py)、[共享工具](filing_tools.py)、[三角色图](multi_graph.py)和[独立评分器](evaluate.py)。

如果修改了评分规则，可以复制已保存的批次，在副本上离线重评：

```powershell
$copy = "results/rescore-$(Get-Date -Format yyyyMMdd-HHmmssfff)"
Copy-Item -LiteralPath results/held-out-final -Destination $copy -Recurse
& $py -X utf8 evaluate.py --run $copy
```

## 打包项目

在项目根目录运行以下命令，即可把两份 README、源码、年报索引、已保存的实验、报告和幻灯片打成一个 ZIP。程序会记录每个文件的 SHA-256 校验值，并在保存前验证整个压缩包。

```powershell
& $py build_submission.py --output .\submission\filing-qa.zip
& $py build_submission.py --verify .\submission\filing-qa.zip
```
