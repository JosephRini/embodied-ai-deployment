# Observability Portal — V0

Read-only eval viewer over `experiments/`. No database, no auth, no
deployment — binds `127.0.0.1` only. Reads the filesystem fresh on every
request, so there's nothing to go stale.

## Run it

```
uv run portal
```

Then open http://127.0.0.1:8792/.

## What it shows

- **Runs** — any `experiments/<run>/` directory containing an `eval_info.json`.
- **Config panel** — best-effort read of `experiments/<run>/checkpoints/*/config.json`
  (prefers a migrated checkpoint with `policy_preprocessor.json` if more than
  one is present). `checkpoints/` is gitignored, so this panel says "not
  found" on a fresh clone until you run an eval yourself.
- **Episodes** — success/fail badge, reward, step count (read from the video's
  frame count via `cv2`, not a separate log), and the rollout video inline.
- **Lesson plan / to-dos** — scans `notes/*.md` for markdown checklist items
  (`- [ ]` / `- [x]`) and displays them read-only. The notes are the source of
  truth; check items off there.
- **Not yet instrumented** — a static, hand-written list (not derived from
  data) naming what this V0 cannot show. It's the Module 4 spec, not a bug
  report.

## How to verify this is telling the truth

Pick one episode and cross-check by hand:

```
python3 -c "import json; d=json.load(open('experiments/000-baseline-rollout/eval_info.json')); print(d['per_task'][0]['metrics']['successes'][1], d['per_task'][0]['metrics']['sum_rewards'][1])"
```

Should print `False 0.0`. The portal's episode-1 card for that run should show
a **FAIL** badge and "reward 0.00" — same underlying numbers, no
transformation in between beyond formatting.

For the config panel: open
`experiments/<run>/checkpoints/<checkpoint>/config.json` directly and compare
`n_obs_steps`, `horizon`, `n_action_steps`, etc. against what the panel shows
— they're read from the exact same file, no caching.
