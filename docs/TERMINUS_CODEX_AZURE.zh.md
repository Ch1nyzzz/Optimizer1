# Terminal-Bench 对比实验运行说明（Codex/Azure × Daytona）

本文档面向**拿到仓库后要在自己机器上跑这个实验的同事**，一步步照做即可。
对应的启动脚本是
[`scripts/launch_terminus_codex_azure_daytona.sh`](../scripts/launch_terminus_codex_azure_daytona.sh)，
英文版说明见 [`TERMINUS_CODEX_AZURE.md`](TERMINUS_CODEX_AZURE.md)。

---

## 0. 这个实验在做什么

在 **Terminal-Bench 2.0**（Terminus-KIRA agent）上做一个 **default vs organized
的对比实验**：

- **proposer（出代码的智能体）** = Codex CLI，通过 **Azure OpenAI** 鉴权。
- **solver（跑题目的基座模型）** = **DeepSeek-V4**，跑在你自己的端点上。
- **rollout（每道题在哪跑）** = **Daytona 云沙箱**，每道题的每次尝试都在一个
  远程沙箱里并行执行。

规模：**20 轮进化**，每轮 **20 道题**（30 道 hard 题里的前 20 道），每道题
**2 次尝试**，也就是每轮 **40 个 trial 全部在 Daytona 上并行**。

两个对照臂，**唯一区别是 organized 臂多了一组 RunStore 工具**，其余完全一致
（两臂都拿到同样的 upstream-2 摘要文件）：

| 臂 | 命令差异 | 含义 |
|---|---|---|
| `default` | `--selection-policy default` | skill 模式 `default`，无 RunStore 工具 |
| `organized` | `--organized --selection-policy default` | skill 模式 `organized-summaries`，生成 `state.md`，注册 RunStore 工具 |

两臂**共用同一个 primed baseline**（iter-0 的 KIRA 种子边界），通过
`--baseline-dir` 复用，种子只评一次。

跑完 20 轮后，每个臂会**自动**把训练集 Pareto 边界上最好的 1 个候选拿到
**完整 test 集（59 道留出题）**上评测。

---

## 1. 你需要准备的东西（前置清单）

| 需要 | 说明 |
|---|---|
| 本仓库 | `git clone` 后用 **`main`** 分支 |
| Python 环境 | 能 `pip install -e .` 安装本项目（会自动带上 `harbor` 等依赖） |
| Node + Codex CLI | `npm install -g @openai/codex`，用较新版本 |
| Azure OpenAI | 一个 GPT-5 codex/推理模型的 **deployment**，记下它的 endpoint 和 key |
| Daytona 账号 | <https://app.daytona.io> 注册，**并发沙箱配额 ≥ 40** |
| DeepSeek-V4 端点 | 一个你能访问的 OpenAI 兼容端点，给 solver 用 |

> 说明：实验里所有题目都在 Daytona 云沙箱里跑，**不需要你本地装 Docker、
> 也不需要本地模型服务器**。本地只需要装好 Python 项目和 Codex CLI。

`DAYTONA_API_KEY` 由实验负责人**私下发给你**（不在仓库里）。

---

## 2. 第一步：装好仓库和依赖

```bash
git clone <仓库地址>
cd Optimizer1
git checkout main

# 按项目惯例安装（会带上 harbor>=0.3 等依赖）
pip install -e .

# 验证 Codex CLI
npm install -g @openai/codex
codex --version
```

仓库里**已经包含**：

- Terminal-Bench 参考工程
  `references/vendor/meta-harness/reference_examples/terminal_bench_2/`
  （proposer 要进化的 `agents/` 就在这里）；
- TB2 任务清单 `data/terminus/tasks.json`（解析 test 集要用，**必须存在**）。

这两样都随 `git pull` 一起下来，无需额外下载。

---

## 3. 第二步：配置 Codex CLI 连 Azure OpenAI

把模板配置拷到位并编辑：

```bash
mkdir -p ~/.codex
cp docs/codex_config.azure.toml ~/.codex/config.toml
$EDITOR ~/.codex/config.toml
```

在 `~/.codex/config.toml` 里改两个地方：

- `base_url` — 改成 `https://你的资源名.openai.azure.com/openai/v1`
  （**结尾的 `/openai/v1` 必须保留**）；
- `model` — 改成你的 Azure **deployment 名称**。

注意：

- Azure 的鉴权是 **API key**（下一步写进 `.env`），**没有交互式
  `codex login`**，也**不支持 Entra ID / Azure AD SSO**。
- 如果你本机已经有一个用 ChatGPT 登录的 Codex 配置、不想被覆盖：把模板拷到
  另一个目录，启动脚本时用 `CODEX_HOME=/那个目录` 指过去即可。

---

## 4. 第三步：填写 `.env`

```bash
cp .env.example .env
$EDITOR .env
```

**这个实验只需要填这 3 个**（其余 `EVAL_*` / `JUDGE_*` 是别的实验用的，
保持占位符不动即可）：

| 变量 | 填什么 |
|---|---|
| `AZURE_OPENAI_API_KEY` | 上一步 `config.toml` 里那个 Azure 资源的 key |
| `DAYTONA_API_KEY` | 实验负责人私下发给你的 Daytona key |
| `DEEPSEEK_API_KEY` | 你的 DeepSeek-V4 端点的 key |

`.env` 已被 `.gitignore` 排除 —— **绝对不要把它提交上去**，本仓库是公开的。

如果你的 DeepSeek-V4 **不是官方 `api.deepseek.com`**，见下面第 9 节，把
solver 指到你自己的端点。

---

## 5. 第四步：启动实验

建议**循序渐进**，别一上来就跑全量（全量会消耗大量 Daytona 配额和 token）：

### 5.1 先做接线自检（不产生任何真实调用）

```bash
DRY_RUN=1 bash scripts/launch_terminus_codex_azure_daytona.sh
```

`DRY_RUN=1` 只检查参数和流程是否通，不调用 Harbor/Daytona/模型。

### 5.2 再跑一个极小规模的冒烟测试

```bash
ITERATIONS=1 LIMIT=2 ROLLOUT_CONCURRENCY=4 \
  bash scripts/launch_terminus_codex_azure_daytona.sh
```

1 轮、2 道题、并发 4 —— 用来确认 Azure / Daytona / DeepSeek 三方真的都能连通。

### 5.3 确认无误后跑全量

```bash
nohup bash scripts/launch_terminus_codex_azure_daytona.sh > /dev/null 2>&1 &
```

为什么用 `nohup`：脚本会先在**前台**跑 baseline prime（这一步较久），然后才
把两个臂用 `setsid` 派生为后台进程。用 `nohup`（或在 `tmux` 里跑）能保证
SSH 断了实验也不挂。

脚本启动后会打印一个 `status:` 文件路径，进度都记在那里。

---

## 6. 怎么看进度

```bash
# 总览：baseline prime / 两个臂的启动状态
tail -f logs/launch_terminus_codex_azure_daytona_*.status

# 某个臂的详细日志
tail -f logs/terminus_kira_codex_azure_daytona_default_train_*.log
tail -f logs/terminus_kira_codex_azure_daytona_organized_train_*.log
```

status 文件里的关键标记：

- `BASELINE_PRIME` / `BASELINE_PRIME_DONE` — 共用 baseline 正在/已完成；
- `START` + `PID` — 某个臂已派生为后台进程；
- `BASELINE_PRIME_FAIL` — baseline 失败，两个臂不会启动，去看对应日志。

---

## 7. 结果在哪里

每个臂一个目录：`runs/terminus_kira_codex_azure_daytona_<臂>_train_<时间戳>/`

| 文件 | 内容 |
|---|---|
| `best_candidates.json` | 训练集 Pareto 边界（最优候选） |
| `evolution_summary.jsonl` | 每轮的累积事件日志 |
| `optimizer_summary.json` | 整个 run 的最终汇总 |
| `test_frontier/` | 自动跑的完整 test 集评测结果 |

---

## 8. 可调参数（环境变量）

启动时在命令前面带上即可，例如
`ITERATIONS=10 ROLLOUT_CONCURRENCY=20 bash scripts/launch_terminus_codex_azure_daytona.sh`。

| 变量 | 默认 | 含义 |
|---|---|---|
| `ARMS` | `default,organized` | 要跑哪些臂 |
| `ITERATIONS` | `20` | 每个臂的进化轮数 |
| `LIMIT` | `20` | 每轮题目数（30 道 hard 里的前 N 道） |
| `ROLLOUT_TRIALS` | `2` | 每道题尝试次数 |
| `ROLLOUT_CONCURRENCY` | `40` | Daytona 最大并发沙箱数 —— **务必 ≤ 你的配额** |
| `CODEX_MODEL` | `gpt-5.1-codex` | 你的 Azure deployment 名称 |
| `CODEX_REASONING_EFFORT` | `high` | Codex proposer 的推理强度 |
| `CODEX_HOME` | 不设 → `~/.codex` | 存放 Azure `config.toml` 的目录 |
| `SOLVER_MODEL` / `SOLVER_BASE_URL` / `SOLVER_API_KEY_ENV` | 官方 DeepSeek API | solver 端点，见第 9 节 |
| `BASELINE_DIR` | `runs/...baseline...` | 复用已 prime 好的 baseline |
| `DRY_RUN` | `0` | `1` = 只做接线自检 |

---

## 9. 把 solver 指到你自己的 DeepSeek-V4 端点

脚本默认指向官方 DeepSeek API
（`openai/deepseek-v4-pro` @ `https://api.deepseek.com/v1`，key 取 `DEEPSEEK_API_KEY`）。
要换成你自己的端点，三个变量一起覆盖：

```bash
SOLVER_MODEL=openai/你的deepseek模型名 \
SOLVER_BASE_URL=https://你的端点/v1 \
SOLVER_API_KEY_ENV=你的KEY变量名 \
  bash scripts/launch_terminus_codex_azure_daytona.sh
```

`SOLVER_API_KEY_ENV` 写的是 **`.env` 里存放 key 的那个变量名**（不是 key 本身）。
比如你在 `.env` 里写了 `MY_DEEPSEEK_KEY=xxx`，就 `SOLVER_API_KEY_ENV=MY_DEEPSEEK_KEY`。

> **`SOLVER_MODEL` 怎么填 —— 重要。** `openai/deepseek-v4-pro` 是一个 **litellm
> 模型 id**，格式为 `<provider>/<模型名>`，**这个默认值很可能要按你们
> DeepSeek-V4-Pro 的接入方式改**：
> - **`openai/` 前缀** —— 让 litellm 走「OpenAI 兼容」适配器，配合
>   `SOLVER_BASE_URL` 打到你的端点。若你们的 DeepSeek-V4-Pro 是经别的 provider
>   接入（litellm 原生 `deepseek/`、或某云厂商的适配器），前缀要换成对应的。
> - **`deepseek-v4-pro` 部分** —— 必须是**你们端点/provider 实际认的模型名**，
>   不同部署叫法不同。
> - 请按你们 DeepSeek-V4-Pro 这个 provider 的文档/部署配置，确认正确的
>   `SOLVER_MODEL`（和 `SOLVER_BASE_URL`）再启动。

---

## 10. 常见问题

**Daytona 配额。** 用 `--terminus-env daytona` 时，并发上限取决于你账号的
**并发沙箱配额**，不是本地 CPU/内存。脚本默认 `ROLLOUT_CONCURRENCY=40`
（20 题 × 2 trial 全并行）。如果你的配额低于 40，把 `ROLLOUT_CONCURRENCY`
设成你的配额值即可，超出的 trial 会排队。**如果 trial 起不来，先查配额，
不要急着怀疑模型推理出问题。** 更多 Daytona 细节见
[`TERMINUS_DAYTONA.md`](TERMINUS_DAYTONA.md)。

**没有 step 硬上限。** Terminal-Bench 不像 SWE-bench 那样有"最多 50 步"的硬
上限；KIRA agent 的 episode 上限继承自 harbor，默认极大，实际靠**每道题的
超时**收尾（`--terminus-agent-timeout-multiplier`，脚本默认 2.0）。

**`trace_similar` 工具（organized 臂）。** organized 臂会注册 RunStore 工具，
其中 `trace_similar` 需要一个 OpenAI 兼容的 embedding 端点。如果你有，就在
`.env` 里设 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `DIFF_EMBEDDING_MODEL`；
不设也没关系 —— 这一个工具会优雅降级，其余 RunStore 工具照常工作。

**绝对不要提交 `.env`。** 本仓库是公开仓库，`.env` 已被 git 忽略，但仍请
确认你没有手动把它加进提交。
