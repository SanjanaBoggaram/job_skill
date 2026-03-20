from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def to_score(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def build_resume_lookup(resume_skills: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Return:
    - skill_score: skill(casefold) -> best score
    - parent_to_child_scores: parent(casefold) -> list of child scores
    """

    skill_score: dict[str, float] = {}
    parent_to_child_scores: dict[str, list[float]] = defaultdict(list)

    for item in resume_skills:
        if not isinstance(item, dict):
            continue
        name = str(item.get("skill", "")).strip()
        if not name:
            continue
        score = to_score(item.get("score", 0))

        key = name.casefold()
        prev = skill_score.get(key)
        if prev is None or score > prev:
            skill_score[key] = score

        parents = [str(p).strip() for p in as_list(item.get("parents", [])) if str(p).strip()]
        for p in parents:
            parent_to_child_scores[p.casefold()].append(score)

    return skill_score, parent_to_child_scores


def compute_gap_skills(
    avg_job_skills: list[dict[str, Any]],
    resume_skills: list[dict[str, Any]],
    debug: bool,
) -> list[dict[str, Any]]:
    resume_score_by_skill, parent_to_child_scores = build_resume_lookup(resume_skills)

    out: list[dict[str, Any]] = []

    for item in avg_job_skills:
        if not isinstance(item, dict):
            continue
        target_skill = str(item.get("skill", "")).strip()
        if not target_skill:
            continue

        target_score = to_score(item.get("score", 0))
        parents = [str(p).strip() for p in as_list(item.get("parents", [])) if str(p).strip()]

        key = target_skill.casefold()

        if key in resume_score_by_skill:
            resume_equiv = resume_score_by_skill[key]
            method = "direct"
        else:
            child_scores = parent_to_child_scores.get(key, [])
            if child_scores:
                resume_equiv = sum(child_scores) / float(len(child_scores))
                method = "children_avg"
            else:
                resume_equiv = 0.0
                method = "missing"

        gap = target_score - resume_equiv  # negative allowed

        obj: dict[str, Any] = {
            "skill": target_skill,
            "parents": parents,
            "score": round(gap, 2),
        }
        if debug:
            obj["target_score"] = round(target_score, 2)
            obj["resume_equivalent_score"] = round(resume_equiv, 2)
            obj["resume_match_method"] = method
        out.append(obj)

    out.sort(key=lambda x: (-(float(x.get("score", 0))), str(x.get("skill", "")).casefold()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a gap_skills.json = average_job_skills - resume_skills, with parent->children averaging fallback."
        )
    )
    parser.add_argument("--resume", default="outputs/resume_skills.json", type=Path)
    parser.add_argument("--average", default="outputs/average_job_skills.json", type=Path)
    parser.add_argument("--out", default="outputs/gap_skills.json", type=Path)
    parser.add_argument("--debug", action="store_true", help="Include debugging fields in output")

    args = parser.parse_args()

    resume = load_json(args.resume)
    avg = load_json(args.average)

    if not isinstance(resume, list):
        raise ValueError("resume skills JSON must be a list")
    if not isinstance(avg, list):
        raise ValueError("average job skills JSON must be a list")

    gap = compute_gap_skills(avg_job_skills=avg, resume_skills=resume, debug=bool(args.debug))
    write_json(args.out, gap)

    print(f"Wrote gap skills: {args.out} (items={len(gap)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
