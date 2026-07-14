"""V0 eval-viewer portal. Reads experiments/ + notes/ off disk on every
request — no database, no auth, localhost only. See README.md in this
directory for the "how to verify this is telling the truth" check.
"""

import json
import re
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from observability.portal.curriculum import MODULES

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = ROOT / "experiments"
NOTES_DIR = ROOT / "notes"

CONFIG_FIELDS = [
    "n_obs_steps",
    "horizon",
    "n_action_steps",
    "num_inference_steps",
    "num_train_timesteps",
]

app = FastAPI(title="Observability Portal (V0)")
app.mount("/media", StaticFiles(directory=EXPERIMENTS_DIR), name="media")
app.mount("/notes", StaticFiles(directory=NOTES_DIR), name="notes")


def _video_url(video_path: str) -> str:
    """eval_info.json stores paths like 'experiments/<run>/videos/...';
    the /media mount serves EXPERIMENTS_DIR at its root, so strip the
    leading 'experiments/' segment."""
    rel = Path(video_path)
    if rel.parts and rel.parts[0] == "experiments":
        rel = Path(*rel.parts[1:])
    return f"/media/{rel.as_posix()}"


def _episode_length(video_path: str) -> int | None:
    abs_path = ROOT / video_path
    if not abs_path.exists():
        return None
    cap = cv2.VideoCapture(str(abs_path))
    try:
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(count) if count > 0 else None
    finally:
        cap.release()


def _find_checkpoint_config(run_dir: Path) -> tuple[str | None, dict | None]:
    """Best-effort: look for a checkpoint dir with config.json under
    experiments/<run>/checkpoints/. Prefers a migrated (post-processor-
    pipeline) checkpoint if more than one is present. Returns
    (checkpoint_dir_name, config_dict) or (None, None) if nothing is found —
    checkpoints/ is gitignored, so a fresh clone won't have this."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None, None
    candidates = [d for d in checkpoints_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if not candidates:
        return None, None
    migrated = [d for d in candidates if (d / "policy_preprocessor.json").exists()]
    chosen = migrated[0] if migrated else candidates[0]
    config = json.loads((chosen / "config.json").read_text())
    return chosen.name, config


def _discover_runs() -> list[Path]:
    if not EXPERIMENTS_DIR.is_dir():
        return []
    return sorted(
        d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and (d / "eval_info.json").exists()
    )


def _run_summary(run_dir: Path) -> dict:
    eval_info = json.loads((run_dir / "eval_info.json").read_text())
    overall = eval_info["overall"]
    checkpoint_name, _ = _find_checkpoint_config(run_dir)
    eval_info_mtime = (run_dir / "eval_info.json").stat().st_mtime
    return {
        "run_id": run_dir.name,
        "date": eval_info_mtime,
        "checkpoint": checkpoint_name,
        "n_episodes": overall["n_episodes"],
        "pc_success": overall["pc_success"],
    }


@app.get("/api/runs")
def list_runs():
    return [_run_summary(d) for d in _discover_runs()]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    run_dir = EXPERIMENTS_DIR / run_id
    eval_info_path = run_dir / "eval_info.json"
    if not eval_info_path.exists():
        return {"error": f"no eval_info.json for run '{run_id}'"}

    eval_info = json.loads(eval_info_path.read_text())
    per_task = eval_info["per_task"][0]["metrics"]
    overall = eval_info["overall"]

    # lerobot-eval caps rendered videos at 10 regardless of n_episodes
    # (max_episodes_rendered in lerobot_eval.py), so video_paths can be
    # shorter than successes/sum_rewards/max_rewards. Iterate over the
    # metrics length, not the video count, or episodes 10+ silently vanish.
    video_paths = per_task["video_paths"]
    episodes = []
    for i in range(len(per_task["successes"])):
        video_path = video_paths[i] if i < len(video_paths) else None
        episodes.append(
            {
                "index": i,
                "success": per_task["successes"][i],
                "sum_reward": per_task["sum_rewards"][i],
                "max_reward": per_task["max_rewards"][i],
                "length_steps": _episode_length(video_path) if video_path else None,
                "video_url": _video_url(video_path) if video_path else None,
            }
        )

    checkpoint_name, config = _find_checkpoint_config(run_dir)
    config_panel = None
    if config is not None:
        config_panel = {k: config.get(k) for k in CONFIG_FIELDS}
        config_panel["image_shape"] = config.get("input_features", {}).get("observation.image", {}).get("shape")
        config_panel["state_shape"] = config.get("input_features", {}).get("observation.state", {}).get("shape")
        config_panel["action_shape"] = config.get("output_features", {}).get("action", {}).get("shape")

    return {
        "run_id": run_id,
        "checkpoint": checkpoint_name,
        "config": config_panel,
        "aggregate": {
            "pc_success": overall["pc_success"],
            "n_episodes": overall["n_episodes"],
            "avg_sum_reward": overall["avg_sum_reward"],
            "avg_max_reward": overall["avg_max_reward"],
        },
        "episodes": episodes,
    }


_CHECKLIST_RE = re.compile(r"^\s*-\s\[([ xX])\]\s+(.*)$")


@app.get("/api/todos")
def list_todos():
    """Scans notes/*.md for markdown checklist items ('- [ ]' / '- [x]').
    Single source of truth stays in the notes; this just surfaces it."""
    if not NOTES_DIR.is_dir():
        return []
    groups = []
    for md_file in sorted(NOTES_DIR.glob("*.md")):
        items = []
        for line in md_file.read_text().splitlines():
            m = _CHECKLIST_RE.match(line)
            if m:
                items.append({"checked": m.group(1).lower() == "x", "text": m.group(2)})
        if items:
            groups.append({"file": md_file.name, "items": items})
    return groups


@app.get("/api/curriculum")
def list_curriculum():
    """Interim module overview — see curriculum.py's docstring. Not derived
    from a formal source yet; that's CURRICULUM.md, still pending."""
    return MODULES


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


def main():
    uvicorn.run(app, host="127.0.0.1", port=8792)


if __name__ == "__main__":
    main()
