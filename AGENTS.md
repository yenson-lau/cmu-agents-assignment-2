# AGENTS.md

## Context

This is the starter repo for Assignment 2 of CMU 11-768 (AI Agents): evaluate a data-visualization agent. I build a VLM-based validator that labels agent runs with error families, write my own visualization tasks, and package them as Harbor environments. The full spec is in [ASSIGNMENT.md](ASSIGNMENT.md).

I am doing this assignment as a side project. I have not enrolled in the course. The grading, submission, quiz, private-set and `AI_USAGE.md` requirements in ASSIGNMENT.md do not apply to me, and I have no course-provided Modal credits: all Modal usage is billed to my own account. I still follow the spec so that my work passes `workflow check-submission`.

## Setup

```bash
uv sync
uv run modal setup
scripts/deploy_validator_model.sh      # validator endpoint (Parts 1-4)
scripts/deploy_generation_models.sh    # three agent models (Part 3)
scripts/deploy_agent_runner.sh         # gateway + mini-swe-agent runner (Part 3); reads GENERATION_* from .env
```

- Python is pinned to 3.12 in `.python-version`.
- Copy `.env.example` to `.env`, append `/v1` to every endpoint URL, and use a Modal proxy token (`<token-id>.<token-secret>`) as the API keys. Never commit `.env`.
- Deploying is free, since every app scales to zero. GPU billing starts on the first request and continues until five idle minutes after the last one. A cold start takes 5-10 minutes (about 12 for an empty GLM cache).
- Harbor runs need Docker running locally.
- Deployment troubleshooting: `infrastructure/README.md`. Harbor errors: `harbor/TROUBLESHOOTING.md`.

## Layout

- `validator/solution.py`: the validator I implement (`validate(run)`). `validator/baseline.py` is the reference example, `validator/prediction.py` defines the output schema, and `validator/runner.py` runs a validator over a directory of runs.
- `artifacts/<run_id>/`: released seed runs (`result.json`, `figure.png`), labeled in `seed_labels.json`. Index by `run_id`, not `task_id`.
- `tasks/<task_id>/`: my authored task descriptors (`task.json`, `inputs/`).
- `runs/<task_id>__<agent>/`: runs generated from my tasks. `labels.json` holds my human labels for them, and `labels.example.json` shows the format.
- `harbor/`: `example/` (worked task plus mutants), `template/`, `preflight.py`, and my packaged tasks under `harbor/tasks/<task_id>/`.
- `workflow/`: CLI (`uv run python -m workflow <command>`) for validate-tasks, generate, package, init-labels, validate-labels, score and check-submission.
- `infrastructure/`: Modal deployments (vLLM serving, agent runner, generation gateway). `output/`: prediction files.

## Rules

- The validator may only call `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` (alias `validator`). Don't swap in or add other models.
- Keep the `validate(run)` signature and the schema in `validator/prediction.py` unchanged.
- Don't hard-code public task IDs, run IDs or labels in the validator.
- Don't modify `artifacts/`, `seed_labels.json`, `harbor/example/` or `harbor/template/`. Change `infrastructure/` and `workflow/` only if I ask.
- Authored tasks may only use matplotlib, numpy, pandas, Pillow, scipy and seaborn.
- After `workflow package` and after generating runs, leave the seeded prompt in `harbor/tasks/<task_id>/instruction.md` unchanged.
- Don't make any git commit without my explicit permission for that specific commit. Permission for an earlier commit, or an offer I didn't answer, doesn't count. The same applies to pushing.

## How to help me

- When an implementation depends on a public API or another transferable specification (e.g. the OpenAI chat-completions format with image inputs, vLLM's OpenAI-compatible server, the Matthews correlation coefficient, Harbor's task format, pytest), point me to the source first. Cite public sources (official docs, specs, API references) before source code such as the installed packages in `.venv`; use source code as a secondary reference, or when no public doc covers the point. Name the specific page or file and what to look for in it. Don't give the answer directly unless I ask for it after reading.
- ASSIGNMENT.md and the READMEs are often terse and assume background they don't explain. When a requirement is dense or ambiguous, first restate it in plain terms: what data is split or changed, where the result ends up, and why each constraint exists. Note where the real intent is only visible from a parameter name, a guard condition or a check in `workflow/`, and cite that location.

## Testing and cost

- Free (local, no model calls): `workflow validate-tasks`, `package`, `init-labels`, `validate-labels`, `score`, `check-submission --skip-harbor-runs`, `harbor/preflight.py`, `uv run pytest` (billable tests are deselected by default).
- Local but slow (Docker, no Modal): `harbor run` with `-a oracle`/`-a nop`/mutants, and the full `check-submission`.
- Billable (Modal GPU time on my account): `validator.runner` (validator endpoint), `workflow generate` (generation GPUs: L4, L40S and H100, plus the runner), and any other request to a deployed endpoint. Ask before running any of these, and prefer small subsets while iterating (`generate --models qwen`, or `validator.runner --artifacts` pointed at a directory with a few runs; `--resume` skips runs already predicted).
- Billable (OpenRouter on my account): `uv run pytest -m openrouter` (calls the pinned OpenRouter models). Ask before running it.
- Deploy scripts also build images on Modal's compute (a few CPU-minutes). Ask before running them too.
- After a crashed billable run, check `uv run modal container list` and stop leftovers with `uv run modal container stop <id>`. To keep endpoints from waking at all, `uv run modal app stop <app>` (`hw2-validator-model`, `hw2-generation-models`, `hw2-agent-runner`, `hw2-generation-gateway`). The `modal` CLI is only installed in `.venv`, so the bare `modal` command works only in an activated venv.
