# Embodied AI Deployment

An applied companion to ["Robot Learning: A Tutorial"](https://huggingface.co/spaces/lerobot/robot-learning-tutorial) (LeRobot), focused on the last mile between an open robot foundation model and a working deployment — entirely in simulation, no physical robot involved.

The tutorial teaches the major robot-learning paradigms: classical control, imitation learning, ACT and Diffusion Policy, and generalist vision-language-action models like SmolVLA and π₀. This repo picks up where that leaves off and asks a more operational question: once a policy is trained, what does it take to serve it, watch it, understand why it fails, and make it better?

Concretely, that means:
- Instrumenting policy inference end-to-end — not just "did it succeed," but what it observed, what it predicted, when, and how stale each action was by the time it executed.
- A replayable rollout timeline that ties camera frames, robot state, predicted vs. executed actions, and latency into one view.
- A simulated customer-style engagement: a task the base model fails at, targeted data collection, fine-tuning, and a before/after acceptance report against a stated spec.

Everything here runs in simulation. Progress, failures, and decisions are logged as I go in `/notes` and `/writeups` — this is a working record, not a finished course.

## Status

🚧 Early. Currently in Module 0 (environment setup) / Module 1 (dataset internals). Check `/notes` for the latest dated entry — that's the most current signal on where things actually stand, more current than this README will usually be.

## Results

_Nothing to report yet. This section fills in as modules produce numbers — each entry will link to the run/config that generated it, not just a claimed figure._

<!-- Results table will look roughly like:
| Module | What | Metric | Result | Run |
|---|---|---|---|---|
| 2 | ACT on PushT | success rate (N=?) | — | — |
| 2 | Diffusion Policy on PushT | success rate (N=?) | — | — |
| 5 | SmolVLA, baseline vs. fine-tuned | success rate under perturbation | — | — |
-->