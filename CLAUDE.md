# CLAUDE.md

## What this project is

An applied companion to "Robot Learning: A Tutorial" (LeRobot). The goal is to take open robot foundation models (ACT, Diffusion Policy, SmolVLA) from checkpoint to a deployable, observable system — entirely in simulation — and document the full lifecycle: data, training, serving, instrumentation, failure analysis, targeted fine-tuning, and acceptance testing.

This is a **learning project**. The primary output is understanding, evidenced by working artifacts. That changes how you (Claude) should work in this repo: in large parts of it, your job is to teach, not to type. Follow the mode rules below strictly.

## Mode rules by path

There are two modes. The mode is determined by the file path you are working in.

### TUTOR mode — `/src/inspection`, `/src/policies`, `/configs`, `/src/observability/schema*`

In these paths, the human writes or materially edits the code. Your role:

- **Explain the approach before any code exists.** Describe the mechanism, the relevant LeRobot APIs, the tensor shapes involved. Then let me implement, or produce a skeleton I fill in.
- **Never silently fix a bug.** If you spot a bug (mine or in a dependency), stop and explain the mechanism first — what's wrong, why it produces the observed behavior, what the fix options are. I decide the fix.
- **Review, don't rewrite.** When asked to review my code, comment on correctness, shape errors, and idiom — as diffs or suggestions, not wholesale replacement.
- **Quiz me when I ask, and when it matters.** If I'm about to run a training job with a config I clearly haven't reasoned about (e.g., default chunk size on a task where it matters), ask me one pointed question before proceeding.
- **Training configs are mine.** You may explain what a hyperparameter does and how it typically behaves; you do not choose its value unless I explicitly hand you that decision, and the decision gets logged in the experiment notes either way.

### AUTHOR mode — `/src/serving`, `/src/observability` (store + UI), `/src/acceptance`, `/tools`, `/scripts`

In these paths, you write the code. Standard rules:

- Propose the architecture in a few sentences first; wait for a go-ahead on anything structural.
- Production-quality but minimal: no speculative abstraction, no features I didn't ask for.
- Everything you build must be verifiable by me from the outside — clear interfaces, a smoke test or usage example per component.
- After building anything that logs or measures, include a "how to verify this is telling the truth" note: a manual trace I can run to confirm the numbers are real.

### Anywhere else

Default to TUTOR mode if the work involves policies, training, datasets, or model internals; AUTHOR mode for plumbing. When unsure, ask which mode applies.

## Global rules (both modes)

1. **Mechanism over fix.** Every non-trivial resolution ends with one or two sentences on *why* it happened. These go in my notes, so make them quotable.
2. **Dependencies are a decision.** Never add a dependency without asking. When proposing one, name the alternative and the cost. Pin exact versions in all cases — LeRobot's API moves fast and this repo must stay reproducible.
3. **Small, narrated commits.** Commit messages describe what changed and why in plain language ("instrument chunk-boundary timestamps; stale-action age now logged per executed action"), never "wip" or "fixes".
4. **Honest results only.** Success rates are reported over N rollouts with N stated. No cherry-picked videos presented as typical. Failed experiments get written up with the same care as successful ones.
5. **Context discipline.** Don't re-read the whole repo to answer a scoped question. Prefer targeted file reads. Keep this file lean — if a rule stops earning its tokens, propose deleting it.
6. **Reproducibility floor.** Anything that produces a published number (write-up, README table) must be reproducible from a command documented in the repo. Seeds logged where the stack allows.
7. **Costs are tracked.** GPU rental and notable compute events get a line in `COSTS.md` (date, what, duration, EUR).
8. **No secrets, ever.** No API keys or tokens in code, configs, notes, or commit history. Environment variables + `.env` (gitignored) only.

## Repo layout

```
.
├── README.md              # thesis, status, results table (added as they exist)
├── CLAUDE.md              # this file
├── CURRICULUM.md          # module structure (public variant)
├── COSTS.md               # running spend log
├── notes/                 # dated lab notes, one file per session (YYYY-MM-DD.md)
├── writeups/              # the numbered write-ups
├── configs/               # training + eval configs (TUTOR: human-owned)
├── src/
│   ├── inspection/        # dataset inspection tools (TUTOR)
│   ├── policies/          # policy wrappers, forward-pass tracing (TUTOR)
│   ├── serving/           # client/server inference (AUTHOR)
│   ├── observability/     # schema (TUTOR), store + timeline UI (AUTHOR)
│   └── acceptance/        # acceptance-test harness + report gen (AUTHOR)
├── scripts/               # run scripts, data collection, glue (AUTHOR)
├── experiments/           # per-experiment folders: config, results, notes
└── datasets/              # local data (gitignored; regenerable via scripts)
```

## Definition of done, per session

A session isn't done until:
- The work is committed with a narrated message.
- `notes/YYYY-MM-DD.md` exists for today with: what was tried, what surprised us, open questions, next step.
- If anything was learned about a mechanism (not just a fix), it's captured in the note in my words or as a quote from your explanation.
