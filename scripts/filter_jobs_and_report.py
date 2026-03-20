from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Keep aligned with scripts/extract_resume_skills.py and scripts/build_job_postings_index.py
OPENROUTER_CHAT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
OPENROUTER_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embed_query(text: str, embedding_model: str) -> list[float]:
    import requests

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

    r = requests.post(
        f"{OPENROUTER_BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": embedding_model, "input": text},
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter embeddings error {r.status_code}: {r.text}")
    data = r.json()
    items = data.get("data")
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Unexpected embeddings response: {data}")
    emb = items[0].get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"Missing embedding in response: {data}")
    return [float(x) for x in emb]


def load_ontology(ontology_path: Path) -> dict[str, list[str]]:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for skill_name, meta in data.items():
        parents = meta.get("parents", [])
        if not isinstance(parents, list):
            parents = []
        out[str(skill_name)] = [str(p) for p in parents]
    return out


def ontology_parents(ontology: dict[str, list[str]], skill: str) -> list[str] | None:
    direct = ontology.get(skill)
    if direct is not None:
        return direct
    s = skill.casefold()
    for k, parents in ontology.items():
        if k.casefold() == s:
            return parents
    return None


def _matches_location(location_text: str, query: str) -> bool:
    if not query:
        return True
    loc = (location_text or "").casefold()
    q = query.casefold().strip()
    if not q or q == "none":
        return True

    # "%like%" behavior: all tokens must appear somewhere in the location string.
    tokens = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    return all(t in loc for t in tokens)


def filter_jobs(
    index_items: list[dict[str, Any]],
    location_query: str,
    role_query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    # Hard filter by location first.
    hard_filtered = [it for it in index_items if _matches_location(str(it.get("location", "")), location_query)]

    if not role_query or role_query.strip().casefold() == "none":
        return hard_filtered[:top_k] if top_k > 0 else hard_filtered

    embedding_model = str(hard_filtered[0].get("embedding_model") or OPENROUTER_EMBED_MODEL)
    qvec = embed_query(role_query, embedding_model=embedding_model)

    scored: list[tuple[float, dict[str, Any]]] = []
    for it in hard_filtered:
        vec = it.get("role_embedding")
        if not isinstance(vec, list):
            continue
        sim = cosine_similarity(qvec, [float(x) for x in vec])
        obj = dict(it)
        obj["role_similarity"] = sim
        scored.append((sim, obj))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [obj for _, obj in scored]
    if top_k > 0:
        filtered = filtered[:top_k]
    return filtered


def average_job_skills(
    postings: list[dict[str, Any]],
    ontology: dict[str, list[str]],
) -> list[dict[str, Any]]:
    score_acc: dict[str, list[int]] = defaultdict(list)
    parents_counter: dict[str, Counter[str]] = defaultdict(Counter)
    casing_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for post in postings:
        skills = post.get("skills", [])
        if not isinstance(skills, list):
            continue
        for s in skills:
            if not isinstance(s, dict):
                continue
            name = str(s.get("skill", "")).strip()
            if not name:
                continue
            key = name.casefold()
            casing_counter[key][name] += 1

            try:
                score = int(s.get("score", 0))
            except Exception:
                score = 0
            score = max(0, min(10, score))
            score_acc[key].append(score)

            onto = ontology_parents(ontology, name)
            if onto is not None:
                for p in onto:
                    parents_counter[key][p] += 1
            else:
                parents_val = s.get("parents", [])
                if isinstance(parents_val, str):
                    parents_counter[key][parents_val] += 1
                elif isinstance(parents_val, list):
                    for p in parents_val:
                        p = str(p).strip()
                        if p:
                            parents_counter[key][p] += 1

    out: list[dict[str, Any]] = []
    for key, scores in score_acc.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        avg_score = int(round(avg))
        display = casing_counter[key].most_common(1)[0][0]
        parents = [p for p, _ in parents_counter[key].most_common()]
        out.append({"skill": display, "parents": parents, "score": max(0, min(10, avg_score))})

    out.sort(key=lambda x: (-int(x["score"]), str(x["skill"]).casefold()))
    return out


def generate_gap_report_with_agent(
    resume_skills: list[dict[str, Any]],
    avg_job_skills: list[dict[str, Any]],
    ontology: dict[str, list[str]],
    model_name: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
    )
    system = (
        "You are a career coach. Compare a candidate's resume skills vs the averaged target job skills. "
        "Produce an actionable skill-gap report. Prioritize skills that are high-scoring in the job average "
        "but missing/low in the resume. Group by parent categories when helpful. "
        "Output Markdown with sections: Strengths, Gaps (High/Medium/Low), Suggested Projects, 30-day Plan. "
        "Be specific and practical (what to learn, what to build)."
    )

    payload = {
        "resume_skills": resume_skills,
        "average_job_skills": avg_job_skills,
        "skill_ontology": ontology,
    }
    human = "Data (JSON):\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    resp = client.chat.completions.create(
        model=(model_name or OPENROUTER_CHAT_MODEL),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": human},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter jobs by location + semantic role, average skills, and generate a gap report.")
    p.add_argument("--index", default="outputs/job_postings_index.json", help="Job postings index JSON")
    p.add_argument("--resume-skills", default="outputs/resume_skills.json", help="Resume skills JSON")
    p.add_argument("--ontology", default="data/skill_ontology.json", help="Ontology JSON")
    p.add_argument("--out-filtered", default="outputs/filtered_job_postings.json", help="Filtered jobs output JSON")
    p.add_argument("--out-average", default="outputs/average_job_skills.json", help="Average job skills output JSON")
    p.add_argument("--out-report", default="outputs/skill_gap_report.md", help="Gap report Markdown output")

    # If provided, avoids prompting.
    p.add_argument("--location", default=None, help="Location filter (or 'none')")
    p.add_argument("--role", default=None, help="Role query (or 'none')")
    p.add_argument("--top-k", type=int, default=5, help="Top K jobs after role similarity")
    return p


def main() -> int:
    args = build_parser().parse_args()

    load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

    model_name = OPENROUTER_CHAT_MODEL

    index_data = load_json(Path(args.index))
    index_items = index_data.get("items", [])
    if not isinstance(index_items, list):
        raise ValueError("Index JSON missing 'items' list")

    location = args.location
    if location is None:
        location = input("Enter job location filter (or 'none'): ").strip()

    role_query = args.role
    if role_query is None:
        role_query = input("Enter job role query for semantic filter (or 'none'): ").strip()

    top_k = int(args.top_k)

    filtered = filter_jobs(index_items, location_query=location or "", role_query=role_query or "", top_k=top_k)
    write_json(Path(args.out_filtered), {"items": filtered})

    ontology = load_ontology(Path(args.ontology))
    avg = average_job_skills(filtered, ontology=ontology)
    write_json(Path(args.out_average), avg)

    resume_skills = load_json(Path(args.resume_skills))
    if not isinstance(resume_skills, list):
        raise ValueError("Resume skills JSON must be an array")

    # report = generate_gap_report_with_agent(resume_skills, avg, ontology=ontology, model_name=model_name)
    # write_text(Path(args.out_report), report)

    print(f"Filtered jobs: {len(filtered)}")
    print(f"Wrote filtered jobs to {args.out_filtered}")
    print(f"Wrote average job skills to {args.out_average}")
    # print(f"Wrote gap report to {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
