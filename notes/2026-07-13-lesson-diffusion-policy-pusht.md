# Lesson: PushT + Diffusion Policy — what's actually happening per step

*First hands-on lesson from Module 2, baseline rollout of `lerobot/diffusion_pusht`.*

## What PushT is

A standard, deliberately minimal robot-learning benchmark. Not real-world, not 3D — a flat 2D surface simulated with pymunk (a lightweight physics engine).

- **Blue circle** — the pusher (the thing the policy controls). It can only push, not grasp.
- **Grey T-shape** — the block being manipulated. Starts in a randomized position/rotation each episode.
- **Green outline** — the target pose the T needs to match, both position and rotation.

Task: push the T-block from wherever it starts into the green target. Success is roughly "how well the final T pose overlaps the target."

Why it's used: cheap to simulate, trivial to visually verify success/failure. The "hello world" of manipulation — nobody cares about PushT itself, it's a fast sandbox for confirming a policy learned *something* before spending real compute on harder tasks (LIBERO, Meta-World, etc. — coming in Module 3).

## How the checkpoint was trained

- **Data:** 206 human teleoperated demonstration episodes. A person controlled the pusher directly and demonstrated the task repeatedly, with variation in starting positions.
- **Paradigm: behavioral cloning.** No reward, no exploration, no RL. Pure imitation — learn to reproduce the mapping from (what was seen, what state we were in) → (what the human did next).
- **This maps directly to the Module 1 formula:** `π(a_t | o_t, s_t, l)` — here, `o_t` = image, `s_t` = pusher state, `l` drops out (PushT has no language conditioning, it's one fixed task).

## Why "Diffusion" — training mechanics

Instead of training the network to output one action directly (regression), Diffusion Policy trains it to **reverse a noising process**:

1. Take a real (observation, action-chunk) pair from a demonstration.
2. Add random noise to the action chunk at some noise level.
3. Train the network to predict the noise that was added — i.e., "given this noisy chunk and this observation, what noise do I need to remove to recover the true action?"
4. Repeat across many noise levels, many demonstrations.

**Why bother, instead of simple regression (what ACT does):** human demonstrations are often multimodal — several equally valid ways to push the block from a given state (nudge left vs. right). A network trained to regress one "best" action directly tends to average across valid options and produce a blurry, invalid compromise. A diffusion-style model represents the whole *distribution* of plausible actions instead of one point estimate — same reason diffusion image models produce sharper, more diverse images than direct pixel regression would.

## Inference — running it backward

At rollout time, the network runs the reverse process: start from pure random noise, iteratively denoise it step by step (conditioned on the current image + state) until what's left is a clean, plausible action chunk.

**This is why our rollout took ~16 minutes instead of the "well under a minute" I first estimated:** every single action chunk requires *dozens* of denoising steps — dozens of full forward passes through the network — not one. On CPU/MPS (no full CUDA), each pass is slow, and this dominated total time completely; the simulator's own physics were cheap and irrelevant to the runtime.

**Contrast for later (Module 2 write-up, ACT vs. Diffusion):** ACT does direct sequence regression — one forward pass, no denoising, no noise-prediction objective. Same imitation-learning umbrella, same demonstration data format, completely different training mechanics and a completely different latency profile at inference. Diffusion Policy trades inference speed for the ability to represent multimodal action distributions.

## Training vs. inference — same conditioning window, different job

The image+state conditioning is identical in both phases: same 96×96×3 image, same (x,y) state, same 2-step history. What differs is what else goes in, and what the network is asked to produce.

**Training, one step:**

| Input | Size |
|---|---|
| Image history (2 steps) | 55,296 |
| Agent state history (2 steps) | 4 |
| Noised action chunk (16 steps × 2) — corrupted version of the *real* demonstrated actions | 32 |
| Noise-level scalar — how much noise was added | 1 |
| **Total input** | **~55,333** |

| Target (what the loss compares against) | Size |
|---|---|
| The actual noise that was added to the chunk | 32 |

The objective: given the real demonstration's action chunk with some random amount of noise mixed in, and told how much noise that was, predict the noise itself — so it can later be subtracted back out. Loss = difference between predicted noise and actual added noise.

**Why this is cheap despite the model being expensive at inference:** training is a *single forward pass per example* — one noise level, one prediction, done. It never has to run the multi-step reverse process during training, because it already has the ground-truth action to noise. The slow, iterative refinement only shows up at *inference*, when there's no ground truth — the model has to bootstrap from pure noise and denoise repeatedly, which is exactly what turned the rollout into 16 minutes instead of under one.

**One consequence worth remembering:** across the 206-episode dataset and many training epochs, the network sees a *different random noise level every example* — from barely-corrupted actions to nearly-pure noise. That's what lets it run the full multi-step reverse process coherently later: it learned to denoise at every point along the spectrum, not just one.

## What's actually "at play" in one inference step

A step is a *window*, not an instant — history in, chunk out.

**Input** (toy numbers, standard Diffusion Policy PushT defaults — verify against this checkpoint's actual `config.json`):

| Stream | Per-timestep size | × 2 obs. timesteps (history) |
|---|---|---|
| Image (96×96×3) | 27,648 | 55,296 |
| Agent (pusher) x,y | 2 | 4 |
| **Total input** | | **~55,300** |

**Output:**

| Stream | Per-timestep size | × 16 timesteps (action chunk) |
|---|---|---|
| Action (target x,y) | 2 | 32 |

Of the 16 predicted future actions, typically only ~8 actually get executed before the sim has moved on enough to warrant a fresh inference call — the sync/async chunk-execution trade-off from the LeRobot tutorial.

**Five conceptually distinct variables in play, dimensionally dominated by pixels:**
1. Current image
2. Previous image (history)
3. Current agent x,y
4. Previous agent x,y
5. The diffusion noise-level/timestep — internal to the denoising process, threaded through every one of the ~dozens of internal forward passes. Has no equivalent in a simple regression policy.
→ Output: 16 future (x,y) action pairs, predicted jointly as one chunk.

**The imbalance worth remembering:** ~55,296 of ~55,300 input numbers are pixels. State and action are a rounding error dimensionally — yet they carry real, load-bearing meaning (unlike an LLM, where nothing analogous to "agent position" exists at all). This is part of why vision encoders dominate these models' parameter budgets.

## Corrections — two imprecisions worth locking in

**"Stable Diffusion noise minimizer" — right mechanism, wrong object.**
Diffusion Policy uses the *same mathematical framework* as Stable Diffusion —
the denoising/noise-prediction process described above. But it isn't Stable
Diffusion, isn't built on Stable Diffusion's weights, and doesn't touch
pixels as its output at all. Stable Diffusion denoises *images* (predicting
noise added to pixel arrays); Diffusion Policy denoises *action sequences* —
the (x,y) pusher commands from the tables above. Same core idea — learn to
reverse a noising process — applied to a completely different kind of data.
"Diffusion" here names a training/inference *technique* the two happen to
share, not a shared model or lineage. Precise phrasing: "a diffusion-based
policy" or "a policy that uses a denoising diffusion process over actions,"
not "a type of Stable Diffusion."

**"Foundation model" — the more important correction.**
`lerobot/diffusion_pusht` is *not* a foundation model in any meaningful
sense. It was trained from scratch on exactly 206 demonstrations of exactly
one task (PushT). No broad pretraining, no generalization beyond this one
task, no transfer capability — swap the task and this checkpoint is simply
inapplicable, the same way an LLM checkpoint isn't interchangeable across
unrelated domains. It's a small, narrow, task-specific policy.

The actual foundation model in the curriculum is **SmolVLA, coming in
Module 3** — trained broadly across many tasks/datasets, with a real
pretrained vision-language backbone underneath, meant to generalize and be
fine-tuned to new tasks. That's also where the flow-matching/diffusion-style
action generation reappears, but this time *inside* a genuine foundation
model architecture — image-language representation → action expert →
chunked continuous output.

**Accurate one-liner for where this stands right now:** *"I'm running a
small, task-specific diffusion-based policy — not a foundation model — that
uses the same denoising technique as image-diffusion models, applied to
robot actions instead of pixels."*

## Open thread — to verify

- [ ] Pull actual `config.json` values for `lerobot/diffusion_pusht`: obs history length (`n_obs_steps`), action chunk length (`horizon` / `n_action_steps`), image resolution, state dimension. Swap real numbers in for the toy defaults above.
- [ ] Watch `eval_episode_1.mp4` (the one failure, reward 0.0) — did it get stuck, overshoot, or oscillate? Raw visual intuition before any instrumentation exists.
- [ ] Rerun eval with N=20–50 episodes before quoting a success rate anywhere public — n=5 is too small to trust (80% could easily be 60% or 100%).
