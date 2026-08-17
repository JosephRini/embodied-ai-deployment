# Curriculum

Take an open robot policy from checkpoint to a deployable, observable
system — entirely in simulation — and prove every claim with evidence,
not a demo video.

The course is five acts. Each one asks a question, ends in a lesson, and
only counts as done when its artifacts exist on disk — see the portal's
course-home view for the live, computed state of each.

## Act 1 — Run someone else's robot

**Question:** What does a downloaded policy actually do, and can you
trust the number it produces?

**Artifacts:** a pinned environment, a baseline evaluation run at N=50,
a lesson on evaluation as measurement, a list of everything the stock
eval output can't show you.

**Lesson:** A number without N is noise.

## Act 2 — Understand what you ran

**Question:** What is the data, and what does training actually do?

**Artifacts:** a dataset-internals lesson, a note on classical-control
vocabulary, a from-scratch training run, an evaluation of that run
compared against the pretrained baseline from Act 1.

**Lesson:** A policy is fit to demonstrations.

## Act 3 — Inherit a foundation model

**Question:** What does it take to get a checkpoint you didn't train
running on your task?

**Artifacts:** a lesson on the model's inference mechanism, a lesson on
the feature-contract problem, a pre-flight script that catches that
problem before a run wastes time on it, a first evaluation on the new
task.

**Lesson:** A checkpoint is a contract, not just weights.

## Act 4 — See it, break it, fix it

**Question:** When it fails a stated threshold, can you diagnose,
collect, fine-tune, and prove the fix?

**Artifacts:** a serving layer, a rollout timeline, a written acceptance
spec, a perturbation experiment that exposes the failure, a targeted
dataset aimed at it, a fine-tuning run, and a before/after report against
the spec.

**Lesson:** You fix what you can measure.

## Act 5 — Tell the story

**Question:** Can the repo be retold as a ten-minute engineering
narrative?

**Artifacts:** a case study, citing only what the rest of the repo can
back up.

**Lesson:** Evidence over narrative.

## Scope, deliberately

- **No RL.** Every policy here is fit to demonstrations (imitation
  learning), not learned by trial and reward. The tradeoff: no reward
  design, no exploration problem, no sim-to-real corrections that
  RL-trained policies need to unlearn on real hardware — at the cost of
  never exceeding what the demonstrations show.
- **No hardware.** Everything runs in simulation. That trades away every
  real-world problem (contact physics, sensor noise, latency budgets
  under real actuators) for repeatability: an eval you can rerun and get
  the same distribution of outcomes.
- **Sim only, not sim-and-then-real.** This project does not attempt a
  sim-to-real transfer. It stops at "does the policy work in the
  simulator it was evaluated in."
- **One environment for the arc that matters.** Acts 1–2 run on PushT —
  the fundamentals (evaluation, training, dataset internals) don't need
  a harder task to teach cleanly, and PushT's official baseline
  checkpoints made a fast start possible. Act 3 onward commits to a
  single environment, Meta-World, for the rest of the arc: the
  foundation-model, fine-tuning, and acceptance-testing work stays on
  one task family end to end, rather than switching benchmarks between
  acts. (LIBERO was considered and declined for this — its dependency is
  Linux-only, which would have made provisioning a GPU box a blocking
  prerequisite before any of Act 3 could start locally.)
