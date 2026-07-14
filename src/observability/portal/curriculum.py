"""Interim curriculum overview, hand-maintained until CURRICULUM.md exists
(Session 3, per the Module 0 kickoff brief). Derived from what's actually
written in README.md and notes/*.md, not invented — cross-check against
those if this ever looks stale."""

MODULES = [
    {
        "id": 0,
        "title": "Environment & setup",
        "status": "done",
        "summary": "Repo skeleton, git identity, uv + pinned LeRobot install, Hub metadata smoke test.",
        "note": "notes/2026-07-13.md",
    },
    {
        "id": 1,
        "title": "Dataset internals",
        "status": "in_progress",
        "summary": "The policy formula π(a_t | o_t, s_t, l), PushT dataset structure, LeRobotDataset metadata.",
        "note": "notes/2026-07-13-lesson-diffusion-policy-pusht.md",
    },
    {
        "id": 2,
        "title": "Policies & baseline rollout",
        "status": "in_progress",
        "summary": "ACT vs. Diffusion Policy mechanics; baseline rollout of lerobot/diffusion_pusht on PushT.",
        "note": "notes/2026-07-13-lesson-diffusion-policy-pusht.md",
    },
    {
        "id": 3,
        "title": "Foundation models (SmolVLA)",
        "status": "not_started",
        "summary": "Generalist VLA policies with a pretrained vision-language backbone; harder tasks (LIBERO, Meta-World).",
        "note": None,
    },
    {
        "id": 4,
        "title": "Observability",
        "status": "seeded",
        "summary": "The \"everything we could not see\" list (latency, chunk boundaries, action staleness) — this portal is its first artifact.",
        "note": "notes/2026-07-13.md",
    },
    {
        "id": 5,
        "title": "Fine-tuning & acceptance testing",
        "status": "not_started",
        "summary": "Simulated customer engagement: targeted data collection, fine-tuning, before/after acceptance report against a stated spec.",
        "note": None,
    },
]
