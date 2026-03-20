from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_CHAT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"  # same as extract_resume_skills.py


@dataclass(frozen=True)
class GapSkill:
    skill: str
    parents: list[str]
    score: float


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def extract_json_value(text: str) -> Any:
    """Best-effort parse for JSON object/array from model output."""
    cleaned = strip_code_fences(text)

    # Direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try array region
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    # Try object region
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("Could not parse JSON from model output")


def get_openrouter_client():
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing in .env")

    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def chat(client, model: str, system: str, user: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return str(resp.choices[0].message.content or "").strip()


def load_gap_skills(path: Path) -> list[GapSkill]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError("gap_skills.json must be a JSON array")

    out: list[GapSkill] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        skill = str(obj.get("skill", "")).strip()
        if not skill:
            continue
        parents_val = obj.get("parents", [])
        if isinstance(parents_val, str):
            parents = [parents_val]
        elif isinstance(parents_val, list):
            parents = [str(p).strip() for p in parents_val if str(p).strip()]
        else:
            parents = []

        try:
            score = float(obj.get("score", 0.0))
        except Exception:
            score = 0.0

        out.append(GapSkill(skill=skill, parents=parents, score=score))

    return out


def load_resources(resources_path: Path) -> dict[str, list[dict[str, Any]]]:
    data = read_json(resources_path)
    if not isinstance(data, dict):
        raise ValueError("resources file must be a JSON object mapping category -> resources")

    out: dict[str, list[dict[str, Any]]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [x for x in v if isinstance(x, dict)]
    return out


def dedupe_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in resources:
        title = str(r.get("title", "")).strip()
        link = str(r.get("link", "")).strip()
        key = (title + "|" + link).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_candidate_resources_for_gap(
    gap: GapSkill,
    resources_by_category: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # If there are resources under the skill itself, include them.
    candidates.extend(resources_by_category.get(gap.skill, []))

    # Include parent category resources.
    for parent in gap.parents:
        candidates.extend(resources_by_category.get(parent, []))

    return dedupe_resources(candidates)


def llm1_gap_summary(client, model: str, positive_gaps: list[GapSkill]) -> str:
    system = "You are an expert career coach."
    payload = [
        {"skill": g.skill, "parents": g.parents, "score": g.score}
        for g in sorted(positive_gaps, key=lambda x: x.score, reverse=True)
    ]
    user = (
        "These are skill gaps (positive score means the job market needs it more than the resume has). "
        "Higher score = higher priority.\n\n"
        "1) Categorize gaps into High / Medium / Low priority.\n"
        "2) For each category, briefly explain why these matter.\n"
        "3) Keep it concise and actionable.\n\n"
        "Gaps JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return chat(client, model=model, system=system, user=user, temperature=0.2)


def llm2_select_resource_ids(
    client,
    model: str,
    gap: GapSkill,
    candidate_resources: list[dict[str, Any]],
) -> list[str]:
    """Return selected resource ids (e.g. ["r0", "r3"])."""

    # Add deterministic IDs to avoid hallucinated resources.
    resources_with_ids: list[dict[str, Any]] = []
    for i, r in enumerate(candidate_resources):
        obj = {"id": f"r{i}"}
        obj.update(r)
        resources_with_ids.append(obj)

    system = "You are a strict JSON-only selector."
    user = (
        "Given a target skill gap and a list of candidate resources (from the parent category), "
        "select ONLY the resource IDs that are directly relevant to the skill. "
        "Return ONLY valid JSON: {\"selected_ids\": [..]}.\n\n"
        "Rules:\n"
        "- Use only ids that appear in the candidate list.\n"
        "- Prefer up to 2 courses + up to 2 projects (max 4 total).\n"
        "- If none match, return {\"selected_ids\": []}.\n\n"
        f"Gap skill: {gap.skill}\n"
        f"Parents: {gap.parents}\n"
        f"Gap score: {gap.score}\n\n"
        "Candidate resources JSON:\n"
        + json.dumps(resources_with_ids, ensure_ascii=False, indent=2)
    )

    raw = chat(client, model=model, system=system, user=user, temperature=0.0)
    parsed = extract_json_value(raw)
    if not isinstance(parsed, dict):
        return []
    ids = parsed.get("selected_ids", [])
    if not isinstance(ids, list):
        return []
    out: list[str] = []
    valid = {r["id"] for r in resources_with_ids}
    for x in ids:
        s = str(x).strip()
        if s in valid:
            out.append(s)
    # de-dup preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped


def llm3_generate_plan(
    client,
    model: str,
    positive_gaps: list[GapSkill],
    selected_resources_by_skill: dict[str, list[dict[str, Any]]],
) -> str:
    system = (
        "You are an expert career coach. Create a learning plan based on skill gaps. "
        "Use the provided resources (courses/projects) when available. "
        "Output Markdown with sections: Overview, Top Priorities, Weekly Plan (4 weeks), Projects, Tracking Checklist."
    )

    gaps_payload = [
        {"skill": g.skill, "parents": g.parents, "score": g.score}
        for g in sorted(positive_gaps, key=lambda x: x.score, reverse=True)
    ]

    user = (
        "Generate a personalized upskilling plan.\n\n"
        "Skill gaps (sorted):\n"
        + json.dumps(gaps_payload, ensure_ascii=False, indent=2)
        + "\n\n"
        "Filtered resources per gap skill (may be empty lists):\n"
        + json.dumps(selected_resources_by_skill, ensure_ascii=False, indent=2)
    )

    return chat(client, model=model, system=system, user=user, temperature=0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate learning plan from gap_skills + resources using OpenRouter.")
    parser.add_argument("--gaps", default="outputs/gap_skills.json", help="Path to gap_skills.json")
    parser.add_argument("--resources", default="data/study_resources/resources", help="Path to resources JSON file")
    parser.add_argument("--out-plan", default="outputs/learning_plan.md", help="Output plan markdown")
    parser.add_argument("--out-summary", default="outputs/gap_summary.md", help="Output gap summary markdown")
    parser.add_argument("--out-selected", default="outputs/gap_selected_resources.json", help="Output selected resources JSON")
    parser.add_argument("--min-gap", type=float, default=0.01, help="Keep only gaps with score > min-gap")
    parser.add_argument("--max-gaps", type=int, default=25, help="Max number of gap skills to process")
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep between LLM2 calls")

    args = parser.parse_args()

    load_dotenv()

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL

    gaps = load_gap_skills(Path(args.gaps))
    positive = [g for g in gaps if g.score > float(args.min_gap)]
    positive.sort(key=lambda x: x.score, reverse=True)
    if args.max_gaps and args.max_gaps > 0:
        positive = positive[: int(args.max_gaps)]

    resources_by_category = load_resources(Path(args.resources))

    client = get_openrouter_client()

    # LLM 1: summarize/prioritize
    summary = llm1_gap_summary(client, model=model, positive_gaps=positive)
    write_text(Path(args.out_summary), summary)

    # LLM 2: per-skill resource filtering (from parents)
    selected_resources_by_skill: dict[str, list[dict[str, Any]]] = {}

    for idx, gap in enumerate(positive, start=1):
        candidates = build_candidate_resources_for_gap(gap, resources_by_category)

        if not candidates:
            selected_resources_by_skill[gap.skill] = []
            continue

        # Use only the first N candidates to keep prompt small.
        candidates = candidates[:80]

        selected_ids = llm2_select_resource_ids(client, model=model, gap=gap, candidate_resources=candidates)

        # Materialize selected resources.
        id_map = {f"r{i}": r for i, r in enumerate(candidates)}
        selected = [id_map[i] for i in selected_ids if i in id_map]

        selected_resources_by_skill[gap.skill] = selected

        print(f"[{idx}/{len(positive)}] selected {len(selected)} resources for {gap.skill}")
        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

    write_json(Path(args.out_selected), selected_resources_by_skill)

    # LLM 3: final plan
    plan = llm3_generate_plan(
        client,
        model=model,
        positive_gaps=positive,
        selected_resources_by_skill=selected_resources_by_skill,
    )
    write_text(Path(args.out_plan), plan)

    print(f"Wrote: {args.out_summary}")
    print(f"Wrote: {args.out_selected}")
    print(f"Wrote: {args.out_plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
