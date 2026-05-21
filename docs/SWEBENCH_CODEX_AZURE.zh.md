# SWE-bench 对比实验运行说明（Codex/Azure × 本机 Docker）

本文档面向**拿到仓库后要在自己机器上跑这个实验的同事**，一步步照做即可。
对应的启动脚本是
[`scripts/launch_swebench_codex_azure.sh`](../scripts/launch_swebench_codex_azure.sh)。

> ⚠️ **先看这条**：SWE-bench 和 Terminal-Bench 不一样 —— Terminal-Bench 把题目
> 甩给 Daytona 云沙箱，同事只要 API key；**SWE-bench 的跑题和评测都在你本机的
> Docker 里跑**，每道题一个数 GB 的容器镜像。**你必须有一台带 Docker、磁盘和
> CPU 都充足的机器。** 这一点脚本无法绕开。

---

## 0. 这个实验在做什么

在 **SWE-bench**（mini-SWE-agent 代码智能体）上做 **default vs organized 对比**：

- **proposer（出代码的智能体）** = Codex CLI，通过 **Azure OpenAI** 鉴权。
- **eval model（mini-SWE-agent 解题时驱动的基座模型）** = 跑在**你自己的远程
  OpenAI 兼容端点**上。
- **评测** = **官方 SWE-bench harness**，每个候选补丁在一个 **per-instance 的
  Docker 容器**里跑测试，**全部在本机**。

规模：**30 轮进化**，每轮在 **30 道 train 题**上优化。两个对照臂，**唯一区别是
organized 臂多了一组 RunStore 工具**，其余完全一致：

| 臂 | 命令差异 | 含义 |
|---|---|---|
| `default` | `--selection-policy default` | skill 模式 `default`，无 RunStore 工具 |
| `organized` | `--organized --selection-policy default` | skill 模式 `organized-summaries`，生成 `state.md`，注册 RunStore 工具 |

两臂**共用同一个 primed baseline**（iter-0 种子边界），通过 `--baseline-dir`
复用。跑完 30 轮后，每个臂会把训练边界上最好的 1 个候选拿到**留出 test 集**上
评测（数据集里带了 470 道 test 题）。

**数据集已在仓库里**：`data/swebench_train_volatile30.json`（30 道 train +
470 道 test），`git pull` 就有，无需单独下载。

---

## 1. 前置清单

| 需要 | 说明 |
|---|---|
| 一台够强的机器 | **Docker 可用**；建议**空闲磁盘 100GB+**、多核 CPU、内存充足 |
| 本仓库 | `git clone` 后用 **`main`** 分支 |
| Python 环境 | 能 `pip install -e .` 安装本项目 |
| Node + Codex CLI | `npm install -g @openai/codex` |
| `uv` / `uvx` | 跑题和评测都通过 `uvx` 调用；`curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Azure OpenAI | 一个 GPT-5 codex/推理模型的 deployment，记下 endpoint 和 key |
| eval 模型端点 | 一个你能访问的远程 OpenAI 兼容端点，给 mini-SWE-agent 当基座模型 |

关于机器资源：官方 SWE-bench Verified 每道题的 Docker 镜像有数 GB。脚本用
`--cache_level instance --clean True`，每道题评完会清掉镜像，所以峰值磁盘由
并发数决定，但仍需大量空闲空间和带宽。`EVAL_WORKERS` 默认 10 = 同时 10 个评测
容器。

SWE-bench Verified 数据集**本体**由官方 harness 自动从 HuggingFace 拉
（`princeton-nlp/SWE-Bench_Verified`），不用手动下。

---

## 2. 第一步：装好仓库和依赖

```bash
git clone <仓库地址>
cd Optimizer1
git checkout main
pip install -e .

# Codex CLI
npm install -g @openai/codex
codex --version

# uv / uvx
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --version

# 确认 Docker 可用
docker info
```

仓库里**已经包含**：mini-SWE-agent 源码工程
（`references/vendor/mini-swe-agent/`，proposer 要进化的对象）和数据集
`data/swebench_train_volatile30.json`。这两样随 `git clone` 一起下来。

---

## 3. 第二步：配置 Codex CLI 连 Azure OpenAI

和 Terminal-Bench 实验完全一样：

```bash
mkdir -p ~/.codex
cp docs/codex_config.azure.toml ~/.codex/config.toml
$EDITOR ~/.codex/config.toml
```

在 `~/.codex/config.toml` 里改：

- `base_url` — `https://你的资源名.openai.azure.com/openai/v1`
  （**结尾的 `/openai/v1` 必须保留**）；
- `model` — 你的 Azure **deployment 名称**。

Azure 鉴权是 API key（下一步写进 `.env`），**没有交互式 `codex login`**。
若不想覆盖本机已有的 Codex 配置，可把模板拷到别处，启动时用
`CODEX_HOME=/那个目录` 指过去。

---

## 4. 第三步：填写 `.env`

```bash
cp .env.example .env
$EDITOR .env
```

**这个实验只需要填这几个**（`DAYTONA_*` / `DEEPSEEK_*` / `JUDGE_*` 是别的
实验用的，保持占位符不动）：

| 变量 | 填什么 |
|---|---|
| `AZURE_OPENAI_API_KEY` | `config.toml` 里那个 Azure 资源的 key |
| `EVAL_BASE_URL` | 你的 eval 模型端点，如 `https://你的端点/v1` |
| `EVAL_MODEL` | 该端点上的模型名 |
| `EVAL_API_KEY` | 该端点的 key（端点不需要鉴权可留占位符） |

`.env` 已被 `.gitignore` 排除 —— **绝对不要提交**，本仓库是公开的。

---

## 5. 第四步：启动实验

**强烈建议循序渐进** —— 全量会消耗大量 token、Docker 资源和时间。

### 5.1 接线自检（不产生真实调用）

```bash
DRY_RUN=1 bash scripts/launch_swebench_codex_azure.sh
```

### 5.2 极小规模冒烟测试（确认 Docker / Azure / eval 端点都通）

```bash
ITERATIONS=1 SWE_LIMIT=2 EVAL_WORKERS=2 TEST_FRONTIER_LIMIT=2 \
  bash scripts/launch_swebench_codex_azure.sh
```

1 轮、2 道题、2 并发、test-frontier 只评 2 题。**这一步会真的拉 Docker 镜像、
真的调模型**，用来验证整条链路。

### 5.3 确认无误后跑全量

```bash
nohup bash scripts/launch_swebench_codex_azure.sh > /dev/null 2>&1 &
```

脚本会先在**前台**做几项 preflight（Docker / uvx / 密钥 / 数据集），然后跑
baseline prime（前台），最后把两个臂用 `setsid` 派生为后台进程。用 `nohup`
（或 `tmux`）保证 SSH 断了实验不挂。

启动后会打印一个 `status:` 文件路径。

---

## 6. 怎么看进度

```bash
# 总览
tail -f logs/launch_swebench_codex_azure_*.status

# 某个臂的详细日志
tail -f logs/swebench_codex_azure_default_train_*.log
tail -f logs/swebench_codex_azure_organized_train_*.log
```

status 文件关键标记：`BASELINE_PRIME` / `BASELINE_PRIME_DONE`、`START` + `PID`、
`BASELINE_PRIME_FAIL`（失败，两个臂不会启动，去看对应日志）。

---

## 7. 结果在哪里

每个臂一个目录：`runs/swebench_codex_azure_<臂>_train_<时间戳>/`

| 文件 | 内容 |
|---|---|
| `best_candidates.json` | 训练集 Pareto 边界（最优候选） |
| `evolution_summary.jsonl` | 每轮的累积事件日志 |
| `optimizer_summary.json` | 整个 run 的最终汇总 |
| `test_frontier/` | 自动跑的留出 test 集评测结果 |

---

## 8. 可调参数（环境变量）

启动时在命令前面带上即可。

| 变量 | 默认 | 含义 |
|---|---|---|
| `ARMS` | `default,organized` | 要跑哪些臂 |
| `ITERATIONS` | `30` | 每个臂的进化轮数 |
| `SWE_LIMIT` | `30` | 每轮的 train 题数 |
| `EVAL_WORKERS` | `10` | 并发评测容器数 —— 按你机器的 CPU/磁盘调 |
| `EVAL_TIMEOUT_S` | `900` | 单题评测超时 |
| `MINISWE_MAX_TOKENS` | `4096` | mini-SWE-agent 单次回复 token 上限 |
| `TEST_FRONTIER_LIMIT` | `0` | 跑完后 test 集评测题数，**`0` = 全部 470 道**（很重，可调小） |
| `CODEX_MODEL` | `gpt-5.1-codex` | 你的 Azure deployment 名称 |
| `CODEX_REASONING_EFFORT` | `high` | Codex proposer 推理强度 |
| `CODEX_HOME` | 不设 → `~/.codex` | 存放 Azure `config.toml` 的目录 |
| `EVAL_BASE_URL` / `EVAL_MODEL` / `EVAL_API_KEY_ENV` | 取自 `.env` | eval 模型端点 |
| `SWE_DATA_PATH` | `data/swebench_train_volatile30.json` | 数据集路径 |
| `BASELINE_DIR` | `runs/...baseline...` | 复用已 prime 好的 baseline |
| `DRY_RUN` | `0` | `1` = 只做接线自检 |

---

## 9. 常见问题

**为什么必须有 Docker。** SWE-bench 的评测用的是官方 harness
（`swebench.harness.run_evaluation`），它给每道题构建/拉取一个独立的 Docker
镜像、在容器里跑该仓库的测试。mini-SWE-agent 跑题阶段也在容器环境里操作仓库。
没有 Docker 这个实验跑不了 —— 这是 SWE-bench 的固有要求，不是脚本能省掉的。

**test-frontier 很重。** 数据集带了 470 道留出 test 题，`TEST_FRONTIER_LIMIT=0`
时跑完会把最好的候选在全部 470 道上评一遍（470 次 agent 跑题 + 470 次 Docker
评测）。机器不够强或只想看个大概，把 `TEST_FRONTIER_LIMIT` 调小（如 `50`）。

**成本提醒。** 经验值每轮约 16–17M token（proposer ~5.5M + 基座模型 ~11M），
30 轮 × 2 臂 很贵。先用 5.2 的冒烟测试确认无误再跑全量。

**不依赖 Together AI。** 旧脚本把基座模型硬连在 Together 上，这个可移植脚本
已改成你自己的 `EVAL_*` 端点，不需要 `TOGETHER_API_KEY`。

**`trace_similar` 工具（organized 臂）。** organized 臂会注册 RunStore 工具，
其中 `trace_similar` 需要一个 OpenAI 兼容的 embedding 端点。有就在 `.env` 里设
`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `DIFF_EMBEDDING_MODEL`；不设也行 ——
这一个工具会优雅降级，其余 RunStore 工具照常工作。

**绝对不要提交 `.env`。** 本仓库公开，`.env` 已被 git 忽略。
