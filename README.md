# embodied-ai-deployment

**From an open robot foundation model checkpoint to a deployment you can defend — in simulation, one notebook at a time.**

This repo is a working log, not a finished tutorial. Each notebook runs standalone in Colab, on a free GPU, and is meant to be read top to bottom: setup cells do the plumbing, marked answer cells are where the actual thinking happens and are filled in as the work is done. Status below reflects the real state of the repo, not a plan.

## Why notebooks

Robot learning has enormous documentation of *how models work* and very little on *what it takes to run one, see inside it, and fix it when it fails*. This repo picks up roughly where the excellent [LeRobot tutorial](https://huggingface.co/docs/lerobot/il_robots) stops: not "how do these architectures work," but "you've been handed a checkpoint you didn't train — now what."

Notebooks, not a framework or a CLI course, because the point is for every intermediate result — a dataset shape, a rendered frame, a loss curve, a failure video — to be something you look at directly, not something a report tells you happened.

## The four parts

| # | Notebook | Question | Status | |
|---|---|---|---|---|
| 1 | **Borrow** — run someone else's robot | What does a downloaded policy actually do, and can you trust the number it produces? | 🟡 in progress | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JosephRini/embodied-ai-deployment/blob/main/notebooks/01_borrow_run_someone_elses_robot.ipynb) |
| 2 | **Build** — train one from scratch | What does it take to fit a policy to demonstrations yourself? | ⚪ not started | — |
| 3 | **Inherit** — adapt a foundation model | What does it take to get a checkpoint you didn't train running on a new task? | ⚪ not started | — |
| 4 | **Prove** — break it, fix it, measure | Can you diagnose a failure, close the gap, and show the fix with numbers? | ⚪ not started | — |

Status is honest, not aspirational: 🟡 means real progress exists but the notebook isn't finished end to end; ⚪ means not started. This table is updated by hand and may lag the actual notebooks — the notebooks themselves are the source of truth.

## What each part actually contains

**1 · Borrow.** Load the PushT dataset directly and inspect one demonstration. Load `lerobot/diffusion_pusht` — a policy trained by someone else — and run it for 50 episodes. Watch a success and a failure inline. Compute a Clopper-Pearson confidence interval by hand, not with a library call. Write, from actually watching the failure, what the run could not show — the seed for the observability work in part 4.

**2 · Build.** Train ACT from scratch on the same task and dataset, watch the loss curve, and compare honestly against part 1's borrowed baseline. This is the only notebook where "trained," not just "evaluated," becomes an honest word to use.

**3 · Inherit.** Load a real VLA foundation model (SmolVLA) and hit the actual problem with inheriting someone else's checkpoint: it declares a fixed set of cameras and a fixed state/action shape, validated before a single weight is used. Diagnosing that contract — not just running the model — is the notebook's point.

**4 · Prove.** Instrument a served policy well enough to see *why* a rollout failed, deliberately break it against a stated acceptance threshold, collect a small amount of targeted data, fine-tune, and report before/after with paired seeds.

## Scope, stated plainly

- Simulation only. No physical robot.
- Classical control and reinforcement learning are acknowledged, not implemented — this repo starts from imitation learning.
- One environment for parts 1–2 (PushT); a different one for parts 3–4 (Meta-World), because no VLA checkpoint targets PushT — this is documented as a real finding, not glossed over.
- Model-internals / interpretability work is explicitly out of scope for the core four parts. If it appears, it will be marked as a clearly separate, optional extension — and treated with real caution: internal signals can correlate with a failure without proving it caused it.

## Notes

Session-by-session working notes live in [`notes/`](notes/), dated, unedited. Lesson write-ups referenced from the notebooks live in [`writeups/lessons/`](writeups/lessons/).

---

*Every number reported anywhere in this repo is stated with its sample size and, where it means anything, a confidence interval. A success rate without an N is not a result.*
