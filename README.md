# 本地通用 RAG 命令行系统

本项目沿用 Chinese-Classical-Literature-RAG 的 C++17 / CMake / llama.cpp 流程：文档读取 → 切分 → 本地嵌入 → 持久化 → 检索 → 本地生成。已经接通真实 GGUF 推理、交互问答和批量评测。它构建的是使用现有模型的 RAG 应用，不训练新的基础模型。

核心代码不包含应急领域规则、题号到答案的映射或验证集答案。更换输入文件即可用于其他知识库。`examples/` 提供设备管理文档示例。

## 构建

需要 CMake 3.20+、C++17 编译器和已有的 CLI11 / nlohmann-json 子模块。本机使用 Visual Studio 2026、CPU 后端。

```powershell
cd D:\design\RAG
git submodule update --init --recursive
cmake -S . -B build -A x64 -DGGML_CUDA=OFF
cmake --build build --config Release --parallel 8
ctest --test-dir build -C Release --output-on-failure
```

默认优先使用 `third_party/llama.cpp`。不存在时自动复用相邻旧项目的源码，并在配置时显示路径。换机器可以显式指定源码位置：

```powershell
cmake -S . -B build -A x64 -DRAG_LLAMA_DIR="D:/dependencies/llama.cpp"
```

Linux 示例：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRAG_LLAMA_DIR=/path/to/llama.cpp
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

无需模型也能构建和验证文档、索引、关键词检索、评测模块：

```powershell
cmake -S . -B build-lexical -A x64 -DRAG_WITH_LLAMA=OFF
cmake --build build-lexical --config Release
```

这个版本只能使用 `--lexical --retrieve-only`，不能生成答案。启用 NVIDIA GPU 需要安装 CUDA 工具链，并重新配置 `-DGGML_CUDA=ON`，同时调整模型 GPU 层数。本机尚未验证 GPU 后端。

## 配置与模型

`config.json` 使用项目内的模型路径；`config.local.json` 直接复用你旧项目的 BGE-small-zh 和 Qwen2.5-1.5B 文件，无需复制或下载。所有示例请在项目根目录执行，相对模型路径以进程工作目录为准。

生成和嵌入模型分别延迟加载。只做关键词检索时不加载模型，建库时不加载生成模型。模型使用 C API 和 RAII 管理资源。生成提示词使用 GGUF 自带模板；遇到不受当前 llama.cpp 支持的模板会明确报错。

`config.gpt-oss.local.json` 提供旧项目 20B 模型的可选路径。Harmony 模板补全 `final` 消息头后再应用答案语法，避免把字母强制写入未完成的助手消息头。本轮题集作答使用 Qwen2.5-1.5B，结果不代表 20B 模型效果。

| 配置 | 作用 |
| --- | --- |
| `document.chunk_size`, `overlap_size` | TXT/Markdown 分块长度和重叠，单位是 Unicode 字符；JSON 文档记录保留完整边界 |
| `models.embedding`, `generation` | 本地 GGUF 文件路径 |
| `models.n_gpu_layers` | CPU 使用 0，GPU 后端可配置卸载层数 |
| `embedding.n_ctx`, `n_batch`, `n_ubatch` | 嵌入上下文和批次；非因果嵌入要求微批能够覆盖批次 |
| `retrieval.top_k`, `candidate_k` | 最终证据数和每条召回通道的候选数 |
| `retrieval.vector_weight` | 向量通道 RRF 权重，默认 0.25；`--lexical` 完全关闭向量通道 |
| `retrieval.option_queries` | 对选择题分别召回题干及各选项证据，不读取答案 |
| `retrieval.context_chars` | 证据字符预算；实际生成还会按模型 token 窗口裁剪 |
| `generation.*` | 生成窗口、批次、输出长度和采样；选择题使用语法约束及确定性采样 |

长 JSON 记录的嵌入按窗口计算并聚合，不直接丢掉后半部分；窗口均值仍可能弱化局部条件。实际生成预算不足时，低排名证据会被裁剪，实际提供的文本保存在 `trace.context_text`。

## 文档输入

支持 UTF-8 TXT、Markdown、JSON，以及含上述文件的目录。TXT/Markdown 识别标题/章节并使用句末和换行边界切分。JSON 接受一个对象、对象数组或 `{"documents": [...]}`：

```json
[
  {"id": "manual-01", "title": "设备借用", "text": "使用人登记编号，经负责人确认后领取。"}
]
```

内容字段支持 `text` / `content` / `rule_text`，记录 ID 支持 `id` / `doc_id` / `rule_id`。空记录跳过并报告，重复 ID 报错，文件和来源记录 ID 一并保留。包含 `question_text`、`answer` 或 `refered_rules` 的记录拒绝入库，防止混入测试题或答案。知识库目录应只放文档；不要把题集与文档混放后直接整目录建库。

## 通用问答示例

```powershell
.\build\Release\RAGCLI.exe --config config.local.json --build examples/handbook.json --db artifacts/demo.json
.\build\Release\RAGCLI.exe --config config.local.json --chat artifacts/demo.json
.\build\Release\RAGCLI.exe --config config.local.json --db artifacts/demo.json --query "设备故障时应该如何处理？"
.\build\Release\RAGCLI.exe --config config.local.json --db artifacts/demo.json --batch examples/questions.json --output artifacts/demo.results.json
```

交互模式输入 `/exit` 退出；每轮独立检索，没有持久会话记忆。自由问答请求中文回答和证据编号，并独立附上检索来源；模型可能漏写正文中的编号。选择题支持 A. / A、等顺序标记，批量模式自动识别为单项选择题。单题使用 `--choice` 可显式启用选项语法约束。当前不支持多选题。

`--build` 和 `--ingest` 等价。多个文档可以跟在该参数后。增加 `--incremental` 可复用 ID、正文、标题和模型配置均未变化的向量：

```powershell
.\build\Release\RAGCLI.exe --config config.local.json --build examples/handbook.json --db artifacts/demo.json --incremental
```

增量建库的输入列表定义完整知识库，未再次提供的文档会从新索引移除。每次提交完整输入列表，而非只提交修改文件。

## 本题集运行

原始题集只保留在本机，不随 GitHub 仓库分发。以下命令需要在项目根目录自行放入 `初赛验证集dev/` 数据目录；无该目录时可先运行 `examples/` 中的通用问答示例。

规则知识库只使用初赛 800 条正式规则与复赛 200 条正式规则。空记录 `977` 被跳过，实际索引 999 条。500 道验证题中 130 道有多条依据，80 道含典型否定词。5,000 道复赛参考题没有标准答案，部分附带引用文本；这些引用不进入模型输入。

```powershell
.\scripts\run_benchmark.ps1
```

默认运行完整 500 题加权混合检索及本地作答，输出 `artifacts/dev.answers.json`。该脚本使用 `--resume`，再次运行会跳过同配置、同问题、同索引的成功记录。修改模型/配置/检索模式或升级程序后应选择新的输出路径，以免混合实验结果；续跑签名不包含程序二进制版本。

本次已按用户要求停止题集运行，保留 346/500 题的现有答案，未继续生成或评测。已完成题目的详细记录在 `artifacts/dev.answers.json`，简洁答案在 `artifacts/dev.answers.json.answers.json`，停止状态在 `artifacts/dev.answers.json.status.json`。下列命令供以后需要时手动运行。

脚本按文档路径和嵌入配置生成独立索引名，防止不同知识库共用同一个文件。也可用 `-Index artifacts/my-index.json` 指定位置。原文修改后使用 `-RebuildIndex` 更新索引。

关键词作答或仅检索评测：

```powershell
.\scripts\run_benchmark.ps1 -Lexical -Output artifacts/dev.lexical-answers.json
.\scripts\run_benchmark.ps1 -RetrieveOnly -Output artifacts/dev.retrieval.json
```

处理无标准答案的复赛参考集：

```powershell
.\scripts\run_benchmark.ps1 -Questions "初赛验证集dev/复赛新增训练参考集.json" -Output artifacts/reference.answers.json
```

该命令实际生成 5,000 题答案，CPU 耗时明显高于检索评测。无标签数据的答案准确率返回 `null`，不能声称全部答对。

结果文件包含每题答案、检索来源、实际模型上下文和耗时。派生文件：

- `*.answers.json`：只有 `question_id` / `answer`。
- `*.report.json`：答案准确率、规则级召回率、找全所有规则的题目比例及失败题号。
- `*.jsonl`：逐题追加并刷新磁盘的检查点；不要手工混合不同实验的记录。

部分运行时，`accuracy` 以完整有标签题集为分母，缺失题视作失败；`completed_accuracy` 只统计已完成的有标签题。`--retrieve-only` 不计算答案准确率。检索指标只在题目有 `rule_id` 或可解析的 `refered_rules` 编号时计算。

答案准确率适用于本题集的单项选择题。若开放问答附有标准答案，目前按字符串精确比较，不提供语义评分。

重新评测已有结果：

```powershell
.\build\Release\RAGCLI.exe --evaluate "初赛验证集dev/初赛验证集dev.json" --output artifacts/dev.answers.json
```

题集审计和实验结果见 [DATASET.md](DATASET.md)、[BENCHMARK.md](BENCHMARK.md)。

## 来源与边界

借鉴 [Ragent](https://github.com/nageoffer/ragent) 的多路检索、RRF 融合、问题拆分和追踪思路，借鉴 [OpenRAG](https://github.com/linagora/openrag) 的组件边界、结构化入库与来源引用思路。实现使用本项目自己的 C++ 模块；没有复制其服务端代码或引入 Java、Ray、Redis 等部署依赖。

当前适合本地小到中等知识库：向量检索为线性扫描，索引是可检查的 JSON 文件。尚未实现 PDF/OCR、DOCX、独立交叉编码器重排、HTTP 服务、权限隔离、多选题、会话记忆或规则冲突的自动优先级裁决。语法约束保证选项格式，不能保证事实判断正确。性能与准确率以实际报告为准。
