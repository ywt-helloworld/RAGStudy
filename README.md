# 本地 RAG（精简版）

直接复用 `Chinese-Classical-Literature-RAG` 的 C++17 代码和 CMake 构建，保持原来的流程：

读取 JSON 文档记录 → 生成向量 → 保存 `.db` → 问题向量 → 原关键词增强检索 → 流式回答。

保留原来的 `AppConfig`、`Chunk`、`VectorDatabase`、`LlamaModelBase`、`EmbeddingEngine`、`GenerationEngine` 和 `MragApp`。建库入口直接读取 JSON，每条记录作为一个检索单元；已删除 `DocumentProcessor`、TXT/Markdown 处理和演示目录。增加 `--test` 单选题批量评测入口，直接复用现有嵌入、检索与生成逻辑，没有额外的评测类、续跑或 RRF 模块。

文件布局：`src/` 放 `.cpp` 实现，`include/` 放项目 `.hpp` 头文件，`third_party/` 放第三方依赖，`test/` 放四份 JSON 知识库和题集。

## 编译

本地项目根目录是 `D:/design/RAG`，CMake 生成的可执行文件名为 `mrag`（Windows 下为 `mrag.exe`）。源码与第三方库均位于本项目目录内。

远程 Linux 下，在解压后的项目根目录执行 CPU 基线编译；不使用 Windows 的 `-A x64`，Release 类型在配置阶段设置：

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build-cpu --parallel 8
```

PPU 编译命令见下方第 4 步，使用单独的 `build-ppu` 目录。所有构建目录在远程生成，不上传本地 CMake 缓存、Windows `.exe` 或 `.lib`。

本地 Windows 编译命令仅供以后需要时使用：

```powershell
cmake -S . -B build -A x64 -DGGML_CUDA=OFF
cmake --build build --config Release --parallel 4
```

`llama.h` 是 llama.cpp 的 C API 头文件，需要配合库实现和 GGML 使用。`third_party/llama.cpp/` 已包含从旧项目复制并校验的库构建源码及许可证，另有下文说明的一处本地 PPU 兼容修补。CMake 直接编译本项目内的依赖，不再引用旧项目的源码目录。上游来源：[llama.cpp](https://github.com/ggml-org/llama.cpp)。

JSON 使用 `third_party/nlohmann/json.hpp`，CLI11 使用 `third_party/CLI11.hpp` 单头文件（2.7.2，源码提交 `eced4d837d0ab57242b1c25d03780d35bf2b0d52`），无需初始化子模块。库只构建推理功能，关闭上游程序、示例和测试。模型权重独立于源码依赖：`config.json` 使用本项目的相对路径，向量模型放在 `models/embedding/`，回答模型放在 `models/llm/`。回答模型复用旧项目的 GGUF，嵌入模型采用用户远程使用的 `bge-m3-q4_k_m.gguf`。按下面步骤放入对应目录。

## 在远程 PPU 实例运行

参考 `E:/study/study2/docs/OPERATIONS.md`：此前使用 `ppu-train:llm` 镜像、单张 PPU-ZW810（96 GiB）和项目持久盘。它验证的是厂商适配的 PyTorch；本项目使用 C++ llama.cpp，尚未在该 PPU 上编译或运行。PPU 的 CUDA 兼容层提供了一条适配路径，但具体 CUDA 编译器、GGML 内核和 GGUF 模型是否兼容，必须在新实例验证。参考 [PPU 软件栈](https://help.aliyun.com/zh/document_detail/2864586.html)、[SDK 环境初始化](https://help.aliyun.com/zh/document_detail/3030340.html)和 [llama.cpp 构建说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。

### 1. 申请实例并准备目录

可以先申请与 study2 相同的单卡镜像，确认它同时提供 PPU 的 C++ 编译工具和 SDK；仅有 PyTorch 运行环境并不足够。使用新实例的 Jupyter 网页终端，以下命令均在 Linux 终端执行。

用户提供的新实例终端输出已确认 `public/RAG` 目录存在，项目公共目录的文件系统为 `CloudStor_fs`、类型为 `storage`。当前部署路径如下；换实例或项目时需重新核对平台说明。代码、模型、数据库和日志都放在这个专用目录。

```bash
export RAG_ROOT=/inspire/store/project/fdu-sae-agent-mapping-prompt-optimization/public/RAG
df -hT "$(dirname "$RAG_ROOT")"
mkdir -p "$RAG_ROOT"
cd "$RAG_ROOT"
```

确认 `df` 显示的是项目存储，而非 `overlay`。study2 的记录表明容器根目录会在实例停止时清理，实例重建后仍需核对平台说明及挂载。

### 2. 上传当前源码、模型和数据

本机旧 `build/` 目录已清理。在 `D:/design/RAG` 的 PowerShell 终端重新生成源码包，输出到项目根目录，无需本地编译或重建 `build/`：

```powershell
tar -czf rag-source.tar.gz CMakeLists.txt config.json README.md .gitignore src include third_party
```

源码包采用当前工作区文件，不依赖 GitHub 上尚未同步的代码；不含本机 Windows 编译产物、模型、题集和日志。通过 Jupyter 文件管理器进入 `RAG_ROOT` 对应的目录，上传以下 5 个文件：

- `D:/design/RAG/rag-source.tar.gz`
- `bge-m3-q4_k_m.gguf`：使用你已获得的模型文件，远程放在 `RAG/models/embedding/`
- `D:/design/Chinese-Classical-Literature-RAG/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `D:/design/RAG/test/knowledge_base1.json`
- `D:/design/RAG/test/test1.json`

回到远程终端整理目录：

```bash
cd "$RAG_ROOT"
tar --keep-old-files -xzf rag-source.tar.gz
mkdir -p models/embedding models/llm test
mv -n bge-m3-q4_k_m.gguf models/embedding/
mv -n qwen2.5-1.5b-instruct-q4_k_m.gguf models/llm/
mv -n knowledge_base1.json test1.json test/
```

如果此前已把两个模型放入 `models/` 根目录，则改用 `mv -n models/bge-m3-q4_k_m.gguf models/embedding/` 和 `mv -n models/qwen2.5-1.5b-instruct-q4_k_m.gguf models/llm/`。程序从项目根目录运行，两个配置字段分别指向这两个 GGUF 文件，而非文件夹。

目录中的 Hugging Face `model.safetensors` 不能直接作为当前 llama.cpp 程序的模型路径。用户提供的日志中，`models/llm/Qwen2.5-1.5B-Instruct/model.safetensors` 已位于 RAG 目录内，无需移动到项目公共目录。当前流程仍使用上面两个已有 GGUF 文件，分别上传到 `models/embedding/` 和 `models/llm/`。

### 3. 检查工具并编译 CPU 基线

```bash
cmake --version
c++ --version
ppu-smi
command -v nvcc
```

需要 CMake 3.20 或更新版本，以及支持 C++17 的编译器。`nvcc` 缺失时，先按平台说明加载已安装的 PPU SDK 的 `envsetup.sh`；SDK 路径以新镜像为准。若镜像没有开发工具，向平台确认开发镜像或 SDK，不直接安装 NVIDIA CUDA 或覆盖厂商 PyTorch。

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build-cpu --parallel 8
./build-cpu/mrag --adjust models.embedding models/embedding/bge-m3-q4_k_m.gguf
./build-cpu/mrag --adjust models.generation models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
./build-cpu/mrag --adjust embedding.n_ctx 1024
./build-cpu/mrag --adjust embedding.n_batch 1024
./build-cpu/mrag --adjust embedding.n_ubatch 1024
./build-cpu/mrag --adjust models.n_gpu_layers 0
./build-cpu/mrag --adjust retrieval.top_k 5
```

先用一条临时记录验证模型加载、向量生成、数据库读写和回答：

```bash
printf '%s\n' '[{"id":"smoke-1","text":"设备借用前必须登记设备编号。"}]' > build-cpu/smoke.json
./build-cpu/mrag --ingest build-cpu/smoke.json build-cpu/smoke.db
./build-cpu/mrag --chat build-cpu/smoke.db
```

提问“设备借用前必须做什么？”，检查回答后输入“退出”。此时只是在远程 CPU 上运行，尚未使用 PPU。

### 4. 尝试编译并验证 PPU 加速

确认 `nvcc --version` 对应镜像提供的 PPU 兼容工具链，并已加载 SDK 环境。采用独立构建目录，保留 CPU 版本。以下参数是待验证的兼容性尝试，关闭部分可选特性，不代表已确认支持该 PPU：

```bash
nvcc --version
cmake -S . -B build-ppu -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=OFF -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_NO_VMM=ON
cmake --build build-ppu --parallel 8
```

配置或编译失败时停止这一阶段，保存报错供检查；不要把关闭 `GGML_CUDA` 后的成功称为 PPU 加速成功。若 CMake 无法探测架构，需要根据当前 PPU SDK 的编译目标配置 `CMAKE_CUDA_ARCHITECTURES`，不能仅凭设备名称猜测 NVIDIA 架构。

用户提供的构建日志已显示 `compiling for ppu001`，但在 `ggml-cuda/topk-moe.cu` 调用 `__isnanf(wt[i])` 时失败：编译器将其解析为 glibc 的 host 函数，设备内核无法调用。本地依赖将这一行替换为公开的 CUDA 设备函数 `isnan(wt[i])`，保留 NaN 检测逻辑；接口参考 [CUDA Math API](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-math-api/group__CUDA__MATH__SINGLE.html)。修补尚未在远程 PPU 编译或运行验证。

如果远程已经解压了旧源码包，可以在 `RAG_ROOT` 中执行下面的单行替换脚本，然后继续增量编译，不需要删除已有构建目录：

```bash
python - <<'PY'
from pathlib import Path
p = Path("third_party/llama.cpp/ggml/src/ggml-cuda/topk-moe.cu")
data = p.read_bytes()
old, new = b"__isnanf(wt[i])", b"isnan(wt[i])"
if data.count(old) == 1 and data.count(new) == 0:
    backup = Path("build-ppu/topk-moe.cu.before-fix")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(data)
    p.write_bytes(data.replace(old, new, 1))
    print("已修补 NaN 检测函数")
elif data.count(old) == 0 and data.count(new) == 1:
    print("修补已经应用")
else:
    raise SystemExit("源码与预期不同，停止修改")
PY
cmake --build build-ppu --parallel 8
```

编译通过后，在另一个网页终端运行 `watch -n 1 ppu-smi`，再在项目终端执行：

```bash
./build-ppu/mrag --adjust models.n_gpu_layers 999
./build-ppu/mrag --ingest build-cpu/smoke.json build-ppu/smoke.db
./build-ppu/mrag --chat build-ppu/smoke.db
```

同样提问并检查回答、日志和设备活动。`n_gpu_layers=999` 表示请求尽量卸载模型层；它本身不能证明加速已生效。只有模型正常运行且设备监控显示该进程参与计算，才能记录 PPU 路径验证通过。小模型运行很短，可以连续提问以便观察；不要用无关的占卡程序模拟利用率。

### 5. 建立测试集 1 的知识库

PPU 小样例通过后：

```bash
./build-ppu/mrag --ingest test/knowledge_base1.json rules.db
./build-ppu/mrag --chat rules.db
```

若只能使用 CPU，则用 `./build-cpu/mrag` 替代，并先将 `models.n_gpu_layers` 设回 0。程序从当前目录读取 `config.json` 并写入 `logs/mrag.log`，后续终端都需要先进入 `RAG_ROOT`；新的 shell 也需要重新加载 PPU SDK 环境。

现在提供建库、交互问答及 `--test` 自动逐题评测。下面的远程运行状态来自用户提供的终端输出，本地助手未连接新实例或占用 study2 的资源；新增评测入口仍需上传并增量编译。

## 使用

命令与原项目一致，`--ingest` 最后一个参数是输出数据库路径：

`main` 使用 CLI11 解析参数；一次只能选择一种操作，缺少参数和未知选项会自动报错。使用 `--help` 或不带参数运行可查看帮助，不需要先加载配置或模型。

```powershell
.\build\Release\mrag.exe --config
.\build\Release\mrag.exe --help
```

本地规则集建库：

```powershell
.\build\Release\mrag.exe --ingest "test/knowledge_base1.json" "test/knowledge_base2.json" rules.db
.\build\Release\mrag.exe --chat rules.db
```

题集只保留在本机，未上传 GitHub。请只把规则文档传给 `--ingest`，不要传题目或答案文件。

修改配置沿用原命令，例如：

```powershell
.\build\Release\mrag.exe --adjust retrieval.top_k 5
.\build\Release\mrag.exe --adjust generation.temperature 0.3
```

## 批量跑题：`--test`

在项目根目录执行，两个参数依次为测试题集和已建好的数据库：

```bash
./build-ppu/mrag --test "test/test1.json" "rules-1024.db"
```

模型只加载一次，每题独立上下文。测试过程在内存中固定 `retrieval.top_k=5`、`generation.temperature=0`，不改写 `config.json`。标准答案和标准规则只用于数据检查及判分，不送入模型。题号重复、答案无效、缺少标准规则或规则不在数据库中时，开始作答前停止并报错。没有答案的 `test2.json` 不能运行这个评测入口。

所有数据写入项目内的 **`test/result` 文件**，远程对应 `RAG/test/result`。文件采用 UTF-8 JSON Lines 格式：第一行是运行信息和实际配置，每题一行包含原始输出、实际提示词、预测答案、标准答案、Top-5 规则编号及分数和四项判定；最后一行是全集汇总。每题完成后立即刷新到磁盘，中断时已完成记录仍保留；没有 `type=summary` 的最后一行就代表尚未完成。再次执行会覆盖旧结果，不自动续跑。

汇总指标 `accuracy`、`hit_at_5`、`all_hit_at_5`、`recall_at_5` 都在 0～1 范围。运行失败或答案格式不合要求的题计入分母，并分别标为 `retrieval_failed`、`generation_failed`、`invalid_answer`。生成失败时仍保留已经成功完成的检索指标。

查看运行信息及汇总：

```bash
head -n 1 test/result | python -m json.tool
tail -n 1 test/result | python -m json.tool
```

### 上传本次更新并增量编译

把本机 `rag-source.tar.gz` 上传到远程 RAG 目录，替换旧源码包。下面只解压源码、头文件和说明，不覆盖远程模型配置及数据库；CMake 构建文件未改，使用原有 `build-ppu` 即可：

```bash
cd /inspire/store/project/fdu-sae-agent-mapping-prompt-optimization/public/RAG
tar -xzf rag-source.tar.gz src include README.md
cmake --build build-ppu --parallel 8
./build-ppu/mrag --help
./build-ppu/mrag --test "test/test1.json" "rules-1024.db"
```

如果你将重建后的库另存为 `rules.db`，命令第二个参数换成 `rules.db`。数据库与当前配置须使用同一个嵌入模型和长度配置；只更新评测入口本身不需要重新建库。

聊天中支持 `/config`、`/adjust <配置项> <值>`；输入 `exit`、`quit` 或 `退出` 结束。

`AppConfig` 必须从完整的 JSON 文件构造；文件缺失、字段缺失或参数无效时抛出异常。没有默认构造和默认配置回退。设置方法返回当前对象指针，支持链式修改：

```cpp
auto config = std::make_unique<AppConfig>("config.json");
config->setGpuLayers(0)
    ->setRetrievalTopK(5)
    ->setTemperature(0.3);
config->saveToFile("config.json");
```

每个参数有独立的 `set` 方法，均返回 `AppConfig*`。例如嵌入与生成模型分别使用 `setEmbModelPath()`、`setGenModelPath()`，上下文参数分别使用 `setEmbNCtx()`、`setGenNCtx()`，不提供合并设置接口。

设置方法只修改内存配置，校验失败时保持原值；需要写回文件时显式调用 `saveToFile()`。JSON 按记录入库，不使用分块长度或重叠长度，相关配置和设置接口已删除。命令行 `--adjust` 和聊天 `/adjust` 会完成修改及保存。`reset/default` 命令已随默认配置移除。

## JSON 数据

接受文档对象、对象数组或 `{"documents": [...]}`。内容字段支持 `text`、`content`、`rule_text`，编号支持 `id`、`doc_id`、`rule_id`。每条记录直接转为一个完整 `Chunk`，来源及编号写入 `metadata`；空记录跳过，重复编号报错。UTF-8 BOM 可正常读取，不根据文件扩展名切换处理程序。

```json
[
  {"id": "manual-01", "title": "设备借用", "text": "登记设备编号，经负责人确认后领取。"}
]
```

当前嵌入模型为用户提供的 BGE-M3 Q4_K_M GGUF，`embedding.n_ctx`、`n_batch`、`n_ubatch` 均设为 1024。BGE-M3 官方支持最长 8192 token（[模型说明](https://huggingface.co/BAAI/bge-m3)）；实际 GGUF 的长度元数据也需要与配置兼容。原代码以 `min(n_ctx, n_batch)` 作为截断限制，所以只提高 `n_ctx` 仍可能被 `n_batch` 限制；非因果嵌入还需要足够的 `n_ubatch`。检索评分、模型提示模板与生成采样继续沿用原项目。

### 将远程嵌入长度从 512 提高到 1024并重新建库

用户已确认远程实际使用 `bge-m3-q4_k_m.gguf`，已报告的最长规则为 897 token，因此先设为 1024。无需重新编译，只在远程项目根目录执行：

```bash
./build-ppu/mrag --adjust models.embedding models/embedding/bge-m3-q4_k_m.gguf
./build-ppu/mrag --adjust embedding.n_ctx 1024
./build-ppu/mrag --adjust embedding.n_batch 1024
./build-ppu/mrag --adjust embedding.n_ubatch 1024
./build-ppu/mrag --config
./build-ppu/mrag --ingest test/knowledge_base1.json rules-1024.db
./build-ppu/mrag --chat rules-1024.db
```

修改配置后，已有数据库中的向量不会自动更新。重新建库写入 `rules-1024.db`，保留旧 `rules.db` 用于对照；后续检索及评测使用新库和同一嵌入模型配置。检查建库输出中是否仍有截断警告；如果存在，按该 GGUF 的上下文上限继续调整三个长度参数并重建，不能仅凭配置改动宣称所有记录已完整嵌入。本地仅更新配置与说明，远程重建结果等待用户执行确认。

`Chunk` 不再分配内部序列号，只保存正文、来源信息和向量。JSON 文档原有的编号仍保留在 `metadata` 中。二进制 `.db` 保存格式已更新为版本 2，旧版本数据库需要通过 `--ingest` 重新建库。

## 题集与 benchmark

本机四份 JSON 的角色如下：

| 文件 | 条数 | 在当前 RAG 中的用途 |
| --- | ---: | --- |
| `knowledge_base1.json` | 800 | 初赛规则知识库，字段为 `rule_id`、`rule_text` |
| `knowledge_base2.json` | 200 | 补充规则知识库，字段同上 |
| `test1.json` | 500 | 有标准答案和依据编号的验证题，用于检查答案准确率及检索召回 |
| `test2.json` | 5,000 | 无标准答案的参考题，部分包含引用信息，无法直接计算答案准确率 |

当前系统没有训练或微调模型权重。名字里的 train/训练参考不等于模型训练集；前两份文件是检索语料，后两份文件是题目。知识库入库时拒绝含题目、标准答案或题目引用字段的记录。

测试集可以作为 benchmark 的评测数据。完整 benchmark 还包括固定的任务定义、数据划分、指标和评测流程；`.json` 是存储格式，不决定它是训练集还是测试集。当前有答案的 dev 集仍按验证集使用，没有另行建立独立最终测试集。

## 测试集 1 评测标准（v1）

本项目把 `test1.json` 的 500 道单选题称为“测试集 1”，沿用它原有的验证集角色。标准答案均为 A/B/C/D；其中 370 题对应 1 条规则、126 题对应 2 条规则、4 题对应 3 条规则。所有标准规则编号均存在于 `knowledge_base1.json` 的 800 条规则中。

### 固定评测流程

仅用 `knowledge_base1.json` 建库，每条规则对应一个检索单元；测试集 1 的基线不合并复赛规则库。每题只把 `question_text`（含选项）交给系统，禁止把标准 `answer` 或标准 `rule_id` 传入检索、提示词或模型。先调用现有 `hybridSearch` 取最终排名前 5 条，再使用同一批结果生成回答，分别判定检索和回答。

评测时固定 `retrieval.top_k=5`，不要与生成采样的 `generation.top_k` 混淆。生成使用 `temperature=0`，每题独立上下文、生成一次。程序在结果中记录题集与数据库路径、实际配置及逐题完整提示词；正式比较不同版本时还应记录输入文件和模型文件哈希、代码版本，哈希不由本入口自动计算。评测过程不修改当前聊天配置。

### 回答正确性

评测提示要求最终答案按 `【最终答案】A` 的形式输出，字母只能是 A/B/C/D，解释可放在前面的 `【分析依据】` 中。判分只比较最终选项，不评判解释文字。

接受仅有一个选项字母的回答，或恰好一个 `【最终答案】` 标记且其后仅有一个选项字母的回答。提取前统一全角字符、字母大小写，忽略首尾空白、标记后的冒号以及选项后单个句号。不从分析段、题干或整段输出中搜索任意 A/B/C/D。多选、冲突结论、无法提取、拒答、空回答和生成失败均记为错误，并另记状态。

`answer_correct = (predicted_answer == answer)`。答案准确率 `Accuracy = 正确题数 / 500`。答案判断与检索判断互相独立；检索未命中但选项答对，仍计答案正确，同时记录检索未命中。生成失败时仍保留已经成功完成的检索指标。

### Top-5 检索命中

对第 i 题，令 `G_i` 为标准 `rule_id` 的去重集合，`R_i` 为最终检索前 5 条结果的规则编号集合。编号统一为去除首尾空白的字符串，精确匹配；例如 `"20"` 不能匹配 `"202"`。保留原始排名和每条结果的检索分数，但命中与否按编号判定，不按分数阈值或正文关键词判定。

不足 5 条时按实际结果计算，不补齐；重复结果不增加命中数量，也不继续读取第 6 条及之后的结果。空结果或检索失败的三项检索指标均为 0。

| 指标 | 单题判定 | 全集汇总 |
| --- | --- | --- |
| 任一命中 `Hit@5` | `G_i` 与 `R_i` 有交集则为 1，否则为 0 | 命中至少一条标准规则的题数 / 500 |
| 全部命中 `AllHit@5` | `G_i` 中所有规则均在 `R_i` 中则为 1，否则为 0 | 全部标准规则命中的题数 / 500 |
| 规则召回 `Recall@5` | 命中标准规则数 / 标准规则总数 | 500 题单题召回率的算术平均，每题权重相同 |

单规则题的这三个指标一致；多规则题必须区分。示例：标准规则为 `["202", "217"]`，Top-5 只含 `"217"`，则 `Hit@5=1`、`AllHit@5=0`、`Recall@5=0.5`。

### 结果记录与汇总

每题保留 `question_id`、标准选项、预测选项、原始模型输出、标准规则编号、按排名排列的 Top-5 规则编号及分数、`answer_correct`、`hit_at_5`、`all_hit_at_5`、`recall_at_5` 和运行状态。这样可以同时检查答案错误和检索遗漏。

最终报告至少包含 `Accuracy`、`Hit@5`、`AllHit@5`、`Recall@5` 及失败/无效输出题数；另列“答案正确/错误 × 任一命中/未命中”的四类题数，定位检索和生成问题。不合并这些指标为一个总分，不预设及格线。

正式全集报告固定以全部 500 题为分母，失败题不剔除。尚未完成的运行只报告进度和已完成题的临时指标，不声称是全集结果；正式汇总时未完成题计为失败。运行前检查题号唯一、标准选项合法、标准规则集合非空且全部存在于知识库；数据校验失败时停止并报告问题，不能静默过滤题目。

## 日志

程序自动创建 `logs/mrag.log`，按 UTF-8 追加写入。每条记录包含本机时间和 `INFO`、`WARN` 或 `ERROR` 等级，写入后立即刷新。

日志记录配置读取和修改、文档处理、建库、数据库读写、模型加载、检索和回答生成，以及 CLI11 参数错误和模型库警告/错误。业务警告与错误同时显示在终端，模型库警告/错误写入文件，回答继续流式显示。日志只记录问答阶段和数量，不保存完整问题、知识库正文或回答内容。

`logs/` 已加入 `.gitignore`。此前的 `artifacts/` 及其中的备份和试建结果已删除。
