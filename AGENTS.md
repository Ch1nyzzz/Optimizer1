# Repository Guidelines

## Project Structure & Module Organization

Optimizer1 is a Python 3.11 `src`-layout package. Core package code lives in
`src/optimizer1/`; memory scaffold implementations are under
`src/optimizer1/scaffolds/`, and small helpers live in `src/optimizer1/utils/`.
Tests are in `tests/` and mirror major modules such as `test_optimizer.py`,
`test_pareto.py`, and `test_scaffolds.py`. CLI and experiment helpers are in
`scripts/`. Configuration examples are in `configs/`. Runtime outputs belong in
`runs/` and `logs/`; do not treat generated run artifacts as source changes.
External reference checkouts are expected under `references/vendor/`.

## Build, Test, and Development Commands

Install the editable package with development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Install source-backed scaffold dependencies when working on mem0, MemGPT, or
MemoryBank integrations:

```bash
python -m pip install -e '.[dev,source]'
scripts/fetch_reference_repos.sh
```

Run the test suite:

```bash
pytest -q
```

Run a quick dry-run optimization smoke test:

```bash
optimizer1 optimize --run-id smoke_opt --iterations 1 --limit 3 --dry-run \
  --scaffold-extra-json @configs/source_memory.example.json
```

## Coding Style & Naming Conventions

Use 4-space indentation, type hints, and small focused functions. Follow the
existing module style: dataclasses for configuration records, snake_case for
functions and variables, PascalCase for classes, and descriptive test names.
Prefer `pathlib.Path` for filesystem work and structured JSON/YAML parsing over
ad hoc string parsing. Keep generated candidate code under run-local
`generated/` directories, not `src/`.

## Testing Guidelines

Pytest is the only configured test framework. Add or update tests beside the
behavior you change, using filenames `tests/test_<feature>.py` and test
functions named `test_<expected_behavior>`. For optimizer, prompt, dynamic
loading, and scaffold changes, include regression tests that exercise file
paths and serialized JSON payloads. Run `pytest -q` before handing off changes.

## Commit & Pull Request Guidelines

The current history uses concise imperative commit subjects, for example
`Add source-backed memory scaffolds`. Keep commits focused and avoid mixing
runtime artifacts with source edits. Pull requests should include a short
summary, the commands run, relevant run IDs or output paths, and linked issues
when applicable. Include screenshots only for UI-facing changes; most changes
should instead include CLI output or JSON artifact paths.

## Security & Configuration Tips

Do not commit secrets, model API keys, local cache paths, or large generated
outputs. Source-backed scaffolds may depend on local model endpoints and vendor
repositories; document non-default paths in the PR description rather than
hardcoding them.

## Paper Editing Workflow (论文修改三段式)

每次修改 `paper/` 下的章节，按以下顺序串行运行三个 skill，前一步的产物作为下一步的输入：

1. **`ml-paper-writing`**：写初稿或做结构性重写。基于研究仓库内容、`docs/PIPELINE.md`、`docs/EXPERIMENT_INSIGHTS.md` 等数据点起草段落，搭好 claim/evidence 骨架，统一术语和引用。
2. **`nature-polishing`**：把 `ml-paper-writing` 的产物按 Nature 系刊写作风格做学术润色，整理段落结构、句间逻辑、动词选择和被动主动比例。
3. **`humanizer`**：在润色后的版本上去 AI 化，移除 em dash 滥用、rule of three、负向并列、空泛归因等典型 AI 写作痕迹，让文本读起来像人写的。

约束：
- 三步必须串行、不可跳步。每步都直接编辑同一份 `paper/sections/*.tex`，保留 LaTeX 命令、`\cite{}`、`\label{}`、数学环境不变。
- 每步开始前重读整个目标 section（不要只看 diff），保证句间衔接和上下文逻辑一致。
- 完成后跑一次 LaTeX 编译做最低限度的健全性检查。

## NeurIPS 2026 Paper Rules

When editing `paper/`, follow the official NeurIPS 2026 Main Track rules and
the local `paper/neurips_2026.tex` template. Do not invent generic appendix
structure.

**Required PDF order for submission:**

1. Main paper content.
2. References.
3. Optional appendices containing supporting textual material, e.g. proofs,
   derivations, extra experimental details, additional results, or extended
   tables.
4. NeurIPS paper checklist.

The main text is limited to 9 content pages, including all figures and tables.
References, optional technical appendices, and the mandatory checklist do not
count as content pages. The submitted PDF must be at most 50 MB. Accepted
camera-ready papers receive one additional content page.

**Appendix naming and placement:**

- Use LaTeX `\appendix` after `\bibliography{...}` / references and before
  `\input{checklist.tex}`.
- The top-level appendix heading should follow the NeurIPS template language:
  `\section{Technical appendices and supplementary material}` or a concise
  descriptive appendix section under that appendix area, such as
  `\section{Additional experimental details}`.
- Do not create a standalone appendix titled only `Additional Related Work` in
  the submission unless it is truly supplementary and clearly nonessential.
  Related work needed to position the contribution must stay in the main paper.
  If overflow related work is needed, put it under a broader technical appendix
  heading as optional background, e.g. `\section{Supplementary related work and
  implementation details}`, and keep the main Related Work self-contained.
- Appendix material is optional reading for reviewers. The main paper must
  stand alone. Do not move critical experiments, assumptions, definitions, or
  claims needed to justify the abstract/introduction into the appendix.
- Do not upload a separate appendix PDF. Textual appendices belong in the same
  PDF. Code/data/videos may be uploaded separately as one anonymized ZIP
  supplement if needed.

**Anonymity and acknowledgments:**

- The initial submission is double blind. Remove author-identifying
  information from the paper, appendix, supplementary ZIP, code links, and
  external URLs. Any linked material must support anonymous browsing.
- Do not include acknowledgments in the anonymized submission. Use the
  NeurIPS `ack` environment only for camera-ready / final versions.
- Cite the authors' own prior work in the third person. For non-public or
  concurrently submitted own work, use anonymized citations and include the
  anonymized paper in supplementary material if necessary.

**Formatting:**

- Use the current `neurips_2026.sty` without changing margins, font sizes, or
  style parameters. Style or page-limit violations can cause desk rejection.
- Use US Letter paper size. Generate PDFs directly with `pdflatex` when
  possible. Avoid Type 3 fonts; prefer `amsfonts`/`\mathbb{}` over packages
  such as `bbold`.
- Figures: captions go below figures, figures are numbered consecutively, and
  artwork must remain legible in color and grayscale.
- Tables: captions go above tables; tables must be centered, legible, and
  publication-quality. Use `booktabs`; avoid vertical rules. If a table is
  wider than the text block, redesign it or use `\resizebox{\linewidth}{!}{...}`
  rather than letting it protrude past the margins.
- Use LaTeX display math environments rather than bare `$$...$$`.

**Checklist, reproducibility, and ethics:**

- Do not remove the NeurIPS checklist; missing checklist is desk-rejection
  risk. Checklist answers are visible to reviewers and should point to paper
  sections or appendix sections when relevant.
- A separate `Limitations` section is encouraged. Claims in the abstract and
  introduction must match the supported scope of experiments/theory.
- Experimental setup details, hyperparameters, data splits, statistical
  variability, compute resources, licenses, and reproducibility instructions
  should be in the main paper when needed to understand the claim; extended
  details may go in the appendix or anonymized supplement.
- If LLMs or agents are an important, original, or non-standard component of
  the method, document their use in the experimental setup or equivalent
  section. Writing/editing-only LLM use does not need declaration, but all
  citations, figures, and claims must be verified.
- Consider negative societal impacts, safeguards, licenses, human-subjects /
  crowdsourcing details, and IRB or equivalent approval where applicable.

**Deadlines and track constraints checked for 2026:**

- Main Track abstract deadline: May 4, 2026 AOE.
- Full paper and supplementary material deadline: May 6, 2026 AOE.
- Author notification: September 24, 2026 AOE.
- Choose the correct track and contribution type before submission. Papers
  cannot be submitted to multiple NeurIPS tracks/types simultaneously, and the
  track/type cannot be switched after submission.

For `claude-kimi`/Kimi proposer runs, read the `sk...` Kimi/Moonshot credential
key from environment variables, not from an interactive login or a mounted
`~/.kimi` directory. If a repo-local `.env` exists, source it before launching
the run: `set -a && source .env && set +a`. When using the Docker proposer
sandbox, explicitly pass the credential variable with `--proposer-docker-env`,
at minimum `KIMI_API_KEY` or `MOONSHOT_API_KEY` as available. Do not ask the
user to restate this credential policy.

**Docker image selection for proposer sandbox:**

- For `--proposer-agent kimi` with docker, always use `--proposer-docker-image
  docker-claude-kimi:latest` (not `docker-claude:latest`). The `docker-claude-kimi`
  image has `claude-kimi` pre-installed and sets `ANTHROPIC_AUTH_TOKEN` from
  `KIMI_API_KEY`, so no `.claude.json` is needed.  Set `--proposer-docker-home /tmp`
  and pass `--proposer-docker-env KIMI_API_KEY`.  Full example:

  ```bash
  python -m optimizer1.cli optimize --locomo \
    --proposer-agent kimi --selection-policy bandit \
    --proposer-sandbox docker \
    --proposer-docker-image docker-claude-kimi:latest \
    --proposer-docker-user 1023:1023 \
    --proposer-docker-home /tmp \
    --proposer-docker-env KIMI_API_KEY \
    ...
  ```

- For `--proposer-agent claude` with docker, use `--proposer-docker-image
  docker-claude:latest` and mount the host Claude credentials read-only:

  ```bash
  --proposer-docker-home /home/yuhan \
  --proposer-docker-mount /data/home/yuhan/.claude:/home/yuhan/.claude:ro \
  --proposer-docker-mount /data/home/yuhan/.claude.json:/home/yuhan/.claude.json:ro \
  ```

- Never use `docker-claude:latest` + a mounted `claude-kimi` binary for kimi runs;
  that image has no `.claude.json` so Claude Code inside the container will fail to
  authenticate.

**Running test evaluations for a completed train run:**

Use `scripts/evaluate_candidate_json.py` with a candidate spec JSON derived from
the train run's `best_candidates.json`:

```python
# build spec JSON (run once to produce runs/<out>/candidate_spec.json)
import json, pathlib
best_cands = json.loads(pathlib.Path('runs/<train_run>/best_candidates.json').read_text())
cands = best_cands if isinstance(best_cands, list) else best_cands.get('candidates', [])
best = max(cands, key=lambda c: c.get('score', c.get('passrate', 0)))
spec = {
    'name': best['scaffold_name'],
    'scaffold_name': 'memgpt_source',
    'candidate_id': '<prefix>_' + best['candidate_id'],
    'top_k': best['config']['top_k'],
    'window': best['config']['window'],
    'extra': best['config']['extra'],
    'source_family': 'memgpt',
}
pathlib.Path('runs/<out>/candidate_spec.json').write_text(json.dumps(spec, indent=2))
```

Then evaluate:

```bash
nohup python3 scripts/evaluate_candidate_json.py \
  --candidate-json runs/<out>/candidate_spec.json \
  --out runs/<out> --split test --eval-workers 128 \
  --model /data/home/yuhan/model_zoo/Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  > logs/<out>.log 2>&1 &
```

For long-running optimizer jobs that must survive the current session, launch
them with `setsid ... > logs/<name>.log 2>&1 < /dev/null &` rather than plain
`nohup`. In this environment, `setsid` is the reliable way to detach the
process so it stays running with `PPID=1`.
