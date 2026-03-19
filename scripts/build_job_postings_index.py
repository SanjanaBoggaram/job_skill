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
from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedSkill:
    skill: str
    parents: list[str]
    score: int


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from model output."""
    cleaned = strip_code_fences(text)
    cleaned = cleaned.strip()
    # Try direct parse first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find the outermost {...} region.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse a JSON object from model output")


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            chunks.append(page_text)
    return "\n".join(chunks)


def read_posting_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def load_ontology(ontology_path: Path) -> dict[str, list[str]]:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    ontology: dict[str, list[str]] = {}
    for skill_name, meta in data.items():
        parents = meta.get("parents", [])
        if not isinstance(parents, list):
            parents = []
        ontology[str(skill_name)] = [str(p) for p in parents]
    return ontology


def _case_insensitive_ontology_lookup(ontology: dict[str, list[str]], skill: str) -> list[str] | None:
    direct = ontology.get(skill)
    if direct is not None:
        return direct
    s = skill.casefold()
    for k, parents in ontology.items():
        if k.casefold() == s:
            return parents
    return None


def normalize_chat_model_name(name: str) -> str:
    """Normalize legacy env values to currently-available Gemini model ids."""
    raw = (name or "").strip()
    if not raw:
        return "models/gemini-2.0-flash"
    if raw.startswith("models/"):
        return raw

    legacy = raw.casefold()
    # Common legacy names that no longer exist under v1beta for this account.
    if legacy in {"gemini-1.5-flash", "gemini-1.5-flash-latest"}:
        return "models/gemini-flash-latest"
    if legacy in {"gemini-1.5-pro", "gemini-1.5-pro-latest"}:
        return "models/gemini-pro-latest"

    return f"models/{raw}"


def llm_extract_job_posting(
    posting_text: str,
    ontology: dict[str, list[str]],
    model_name: str,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    ontology_json = json.dumps(ontology, ensure_ascii=False)

    system = (
        "Extract structured information from job postings. "
        "Return ONLY valid JSON (no markdown) with keys: company_name, role, location, employment_type, skills. "
        "skills must be an array of objects: {skill: string, parents: string[], score: int 0-10}. "
        "Use this ontology (JSON mapping skill->parents) to choose parents whenever applicable:\n"
        f"{ontology_json}\n\n"
        "If a skill is not in the ontology, choose the closest parent categories from the ontology, or [] if none. "
        "Score guidance: 9-10 required/central; 7-8 strongly preferred; 5-6 mentioned once/as plus; 1-4 tangential."
    )

    human = f"Job posting text:\n\n{posting_text}\n\nReturn only JSON."

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            content = getattr(msg, "content", None)
            return extract_json_object(str(content if content is not None else msg))
        except Exception as e:  # pragma: no cover
            last_err = e
            human = (
                "IMPORTANT: Output must be ONLY valid JSON. No markdown, no extra text.\n\n" + human
            )
            time.sleep(0.75 * (attempt + 1))
    assert last_err is not None
    raise last_err


def embed_text(text: str, embedding_model: str) -> list[float]:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    emb = GoogleGenerativeAIEmbeddings(model=embedding_model)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            v = emb.embed_query(text)
            return [float(x) for x in v]
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(0.75 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _coerce_skills(skills_raw: Any, ontology: dict[str, list[str]]) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    if not isinstance(skills_raw, list):
        return skills

    for s in skills_raw:
        if not isinstance(s, dict):
            continue
        name = str(s.get("skill", "")).strip()
        if not name:
            continue

        parents_val = s.get("parents", [])
        if isinstance(parents_val, str):
            parents = [parents_val]
        elif isinstance(parents_val, list):
            parents = [str(p).strip() for p in parents_val if str(p).strip()]
        else:
            parents = []

        # Overwrite parents with ontology if it exists (keeps taxonomy consistent).
        onto_parents = _case_insensitive_ontology_lookup(ontology, name)
        if onto_parents is not None:
            parents = onto_parents

        try:
            score = int(s.get("score", 0))
        except Exception:
            score = 0
        score = max(0, min(10, score))

        skills.append({"skill": name, "parents": parents, "score": score})

    # De-dup by skill name (case-insensitive), keep the highest score.
    best: dict[str, dict[str, Any]] = {}
    for item in skills:
        key = str(item["skill"]).casefold()
        prev = best.get(key)
        if prev is None or int(item.get("score", 0)) > int(prev.get("score", 0)):
            best[key] = item

    return sorted(best.values(), key=lambda x: (-int(x.get("score", 0)), str(x.get("skill", "")).casefold()))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build outputs/job_postings_index.json with skills + role embeddings.")
    parser.add_argument("--in-dir", default="data/job_postings", help="Directory containing job posting files")
    parser.add_argument("--ontology", default="data/skill_ontology.json", help="Skill ontology JSON")
    parser.add_argument("--out", default="outputs/job_postings_index.json", help="Output index JSON path")

    args = parser.parse_args()

    load_dotenv()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in .env")

    model_name = normalize_chat_model_name(os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash"))
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

    in_dir = Path(args.in_dir)
    ontology = load_ontology(Path(args.ontology))

    items: list[dict[str, Any]] = []
    for path in sorted([p for p in in_dir.iterdir() if p.is_file()]):
        job_id = path.name
        posting_text = _normalize_whitespace(read_posting_text(path))

        extracted = llm_extract_job_posting(posting_text, ontology, model_name=model_name)
        time.sleep(3)
        company_name = str(extracted.get("company_name", "")).strip()
        role = str(extracted.get("role", "")).strip()
        location = str(extracted.get("location", "")).strip()
        employment_type = str(extracted.get("employment_type", "")).strip()
        skills = _coerce_skills(extracted.get("skills", []), ontology)

        role_embedding = embed_text(role or job_id, embedding_model=embedding_model)

        items.append(
            {
                "job_id": job_id,
                "company_name": company_name,
                "role": role,
                "location": location,
                "employment_type": employment_type,
                "skills": skills,
                "role_embedding": role_embedding,
                "embedding_model": embedding_model,
                "source_path": str(path.as_posix()),
            }
        )

    out_path = Path(args.out)
    write_json(out_path, {"items": items})
    print(f"Built index for {len(items)} job postings")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
