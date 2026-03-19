from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text)
    return "\n".join(chunks)


def read_job_posting_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)

    # Most of the synthetic dataset looks like plaintext files without extensions.
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
    lowered = skill.casefold()
    for k, parents in ontology.items():
        if k.casefold() == lowered:
            return parents
    return None


def heuristic_parse_metadata(posting_text: str) -> dict[str, str]:
    lines = [ln.strip() for ln in posting_text.splitlines()]
    lines = [ln for ln in lines if ln and ln != "\u00a0"]

    def looks_like_company(s: str) -> bool:
        s2 = s.casefold()
        return bool(
            re.search(r"\b(pvt\.?|ltd\.?|inc\.?|llc|corp\.?|co\.?|private|limited)\b", s2)
            or any(k in s2 for k in ["technologies", "technology", "solutions", "software", "services"])
        )

    def looks_like_role(s: str) -> bool:
        s2 = s.casefold()
        return any(
            k in s2
            for k in [
                "engineer",
                "developer",
                "intern",
                "scientist",
                "specialist",
                "tester",
                "qa",
                "manager",
                "analyst",
                "trainee",
                "fresher",
            ]
        )

    def looks_like_location(s: str) -> bool:
        s2 = s.casefold()
        if any(k in s2 for k in ["remote", "hybrid"]):
            return True
        if "," in s:
            return True
        # Single tokens like "Bangalore" show up in the dataset.
        if 2 <= len(s) <= 30 and " " not in s and s[0].isalpha():
            return True
        return False

    role = lines[0] if lines else ""
    # Remove trailing markers like "- job post".
    role = re.sub(r"\s*[-|]\s*job\s*post\s*$", "", role, flags=re.I).strip()

    company_name = lines[1] if len(lines) > 1 else ""

    location = lines[2] if len(lines) > 2 else ""

    # Handle cases where the second line repeats the role and company appears later.
    if company_name and looks_like_role(company_name) and len(lines) > 2 and looks_like_company(lines[2]):
        company_name = lines[2]
        if len(lines) > 3 and looks_like_location(lines[3]):
            location = lines[3]

    # Prefer explicit structured fields when present.
    m = re.search(r"(?im)^Company\s*:\s*(.+)$", posting_text)
    if m:
        company_name = m.group(1).strip()

    # Job Location / Location fields.
    m = re.search(r"(?im)^Job\s*Location\s*:\s*(.+)$", posting_text)
    if m:
        location = m.group(1).strip()
    else:
        m = re.search(r"(?im)^Location\s*:\s*(.+)$", posting_text)
        if m:
            location = m.group(1).strip()
        else:
            # The synthetic posts often have a "Location" section header.
            m = re.search(r"(?im)^Location\s*$\s*(.+)$", posting_text)
            if m:
                location = m.group(1).strip()

    # Employment type.
    employment_type = ""
    m = re.search(r"(?im)^Job\s*Type\s*:\s*(.+)$", posting_text)
    if m:
        employment_type = m.group(1).strip()
    else:
        # Find the Job type section and take following non-empty lines until a blank-ish separator.
        parts = re.split(r"(?im)^Job\s*type\s*$", posting_text)
        if len(parts) > 1:
            after = parts[1]
            cand_lines: list[str] = []
            for ln in after.splitlines():
                ln = ln.strip()
                if not ln or ln in {"&nbsp;", "Job details"}:
                    if cand_lines:
                        break
                    continue
                # Stop at other section headers.
                if ln.lower() in {"location", "full job description", "benefits", "pay"}:
                    break
                cand_lines.append(ln)
                if len(cand_lines) >= 3:
                    break
            if cand_lines:
                employment_type = ", ".join(dict.fromkeys(cand_lines))

    # Normalize a bit.
    return {
        "role": _normalize_whitespace(role),
        "company_name": _normalize_whitespace(company_name),
        "location": _normalize_whitespace(location),
        "employment_type": _normalize_whitespace(employment_type),
    }


def heuristic_extract_skills(posting_text: str, ontology: dict[str, list[str]]) -> list[ExtractedSkill]:
    text = posting_text

    found: dict[str, int] = {}
    for skill in ontology.keys():
        escaped = re.escape(skill)
        pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        hits = len(pattern.findall(text))
        if hits > 0:
            found[skill] = hits

    # Add common terms (not necessarily in ontology).
    common_terms = [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "Golang",
        "C++",
        "C#",
        "HTML",
        "CSS",
        "React",
        "Angular",
        "Node",
        "Node.js",
        "Spring",
        "Spring Boot",
        "REST",
        "RESTful",
        "API",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "Docker",
        "Kubernetes",
        "AWS",
        "GCP",
        "Azure",
        "DevOps",
        "CI/CD",
        "Testing",
        "QA",
        "JIRA",
        "Postman",
    ]

    for term in common_terms:
        escaped = re.escape(term)
        pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        hits = len(pattern.findall(text))
        if hits > 0:
            found.setdefault(term, hits)

    def infer_parents(skill: str) -> list[str]:
        parents = _case_insensitive_ontology_lookup(ontology, skill)
        if parents is not None:
            return parents

        s = skill.casefold()
        if s in {"python", "java", "golang", "c++", "c#", "javascript", "typescript"}:
            return ["Programming Fundamentals"]
        if s in {"html", "css", "react", "angular"}:
            return ["Frontend"]
        if s in {"node", "node.js", "spring", "spring boot", "rest", "restful", "api"}:
            return ["Backend"]
        if s in {"aws", "gcp", "azure"}:
            return ["Cloud"]
        if s in {"mysql", "postgresql", "mongodb"}:
            return ["Database"]
        if s in {"docker", "kubernetes", "devops", "ci/cd"}:
            return ["DevOps"]
        if s in {"testing", "qa", "jira", "postman"}:
            return ["Testing"]
        return []

    def score_from_hits(hits: int) -> int:
        base = 4
        bonus = min(6, hits * 2)
        return max(0, min(10, base + bonus))

    items: list[ExtractedSkill] = []
    for skill, hits in sorted(found.items(), key=lambda kv: (-kv[1], kv[0].casefold())):
        items.append(ExtractedSkill(skill=skill, parents=infer_parents(skill), score=score_from_hits(hits)))

    return items[:50]


def llm_extract_job_posting(
    posting_text: str,
    ontology: dict[str, list[str]],
    model_name: str,
) -> dict[str, Any]:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI

    @tool
    def get_skill_ontology() -> str:
        """Return the full skill ontology as JSON mapping skill -> parents."""
        return json.dumps(ontology, ensure_ascii=False)

    system = (
        "You are an information extraction agent for job postings. "
        "Given a raw job posting text, extract: company_name, role, location, employment_type. "
        "Also extract a skills array where each element is: {skill: string, parents: string[], score: int 0-10}. "
        "Use the ontology tool to map skills to parents where applicable; for skills not in ontology, "
        "choose the closest parent categories from the ontology (or [] if none). "
        "Score guidance: 9-10 if clearly required / central to many responsibilities; "
        "7-8 if strongly preferred or appears multiple times; 5-6 if mentioned once or as a plus; "
        "1-4 if only tangential. "
        "Return ONLY valid JSON object with keys: company_name, role, location, employment_type, skills."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Job posting text:\n\n{posting_text}\n\nReturn only JSON.",
            ),
        ]
    )

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    tools = [get_skill_ontology]
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    result = executor.invoke({"posting_text": posting_text})
    raw = strip_code_fences(str(result.get("output", "")))
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output was not a JSON object")
    return parsed


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    emb = GoogleGenerativeAIEmbeddings(model=model_name)
    vectors = emb.embed_documents(texts)
    return [list(map(float, v)) for v in vectors]


def embed_query(text: str, model_name: str) -> list[float]:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    emb = GoogleGenerativeAIEmbeddings(model=model_name)
    v = emb.embed_query(text)
    return list(map(float, v))


def cmd_extract(args: argparse.Namespace) -> int:
    load_dotenv()

    in_dir = Path(args.in_dir)
    ontology = load_ontology(Path(args.ontology))
    out_dir = Path(args.out_dir)
    index_path = Path(args.index_out)

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    posting_paths = sorted([p for p in in_dir.iterdir() if p.is_file()])

    index_items: list[dict[str, Any]] = []

    # First pass: extract metadata + skills.
    role_texts_for_embedding: list[str] = []
    index_positions_for_embedding: list[int] = []

    for path in posting_paths:
        job_id = path.name
        raw_text = _normalize_whitespace(read_job_posting_text(path))

        if args.use_llm:
            if not google_key:
                raise RuntimeError("--use-llm set but GOOGLE_API_KEY is missing in .env")
            extracted = llm_extract_job_posting(raw_text, ontology, model_name=model_name)

            company_name = str(extracted.get("company_name", "")).strip()
            role = str(extracted.get("role", "")).strip()
            location = str(extracted.get("location", "")).strip()
            employment_type = str(extracted.get("employment_type", "")).strip()

            skills_raw = extracted.get("skills", [])
            skills: list[dict[str, Any]] = []
            if isinstance(skills_raw, list):
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
                        parents = [str(p) for p in parents_val if str(p).strip()]
                    else:
                        parents = []
                    try:
                        score = int(s.get("score", 0))
                    except Exception:
                        score = 0
                    score = max(0, min(10, score))
                    skills.append({"skill": name, "parents": parents, "score": score})
            skills = sorted(skills, key=lambda x: (-int(x.get("score", 0)), str(x.get("skill", "")).casefold()))
        else:
            meta = heuristic_parse_metadata(raw_text)
            company_name = meta["company_name"]
            role = meta["role"]
            location = meta["location"]
            employment_type = meta["employment_type"]
            skills = [
                {"skill": s.skill, "parents": s.parents, "score": s.score}
                for s in heuristic_extract_skills(raw_text, ontology)
            ]

        posting_obj: dict[str, Any] = {
            "job_id": job_id,
            "company_name": company_name,
            "role": role,
            "location": location,
            "employment_type": employment_type,
            "skills": skills,
            "source_path": str(path.as_posix()),
        }

        out_path = out_dir / f"{job_id}.json"
        write_json(out_path, posting_obj)

        index_item = {
            "job_id": job_id,
            "company_name": company_name,
            "role": role,
            "location": location,
            "employment_type": employment_type,
            "json_path": str(out_path.as_posix()),
            "source_path": str(path.as_posix()),
        }

        index_items.append(index_item)

        if args.embed_roles:
            role_texts_for_embedding.append(role or job_id)
            index_positions_for_embedding.append(len(index_items) - 1)

    # Second pass: add embeddings if requested and possible.
    if args.embed_roles:
        if not google_key:
            raise RuntimeError("--embed-roles set but GOOGLE_API_KEY is missing in .env")
        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        vectors = embed_texts(role_texts_for_embedding, model_name=embedding_model)
        for idx, vec in zip(index_positions_for_embedding, vectors):
            index_items[idx]["role_embedding"] = vec
            index_items[idx]["embedding_model"] = embedding_model

    write_json(index_path, {"items": index_items})

    print(f"Extracted {len(index_items)} job postings")
    print(f"Wrote index to {index_path}")
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    load_dotenv()

    index_path = Path(args.index)
    data = json.loads(index_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Index file missing 'items' list")

    location_query = (args.location or "").strip()
    role_query = (args.role_query or "").strip()

    filtered = items

    if location_query:
        q = location_query.casefold()
        filtered = [
            it
            for it in filtered
            if q in str(it.get("location", "")).casefold()
        ]

    role_scores: dict[str, float] = {}

    if role_query:
        # Prefer embeddings if available in index; else fallback to token overlap.
        have_embeddings = any(isinstance(it.get("role_embedding"), list) for it in filtered)

        if have_embeddings:
            google_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if not google_key:
                raise RuntimeError("Role embeddings present/needed but GOOGLE_API_KEY missing in .env")
            embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
            qvec = embed_query(role_query, model_name=embedding_model)
            scored: list[tuple[float, dict[str, Any]]] = []
            for it in filtered:
                vec = it.get("role_embedding")
                if not isinstance(vec, list):
                    continue
                sim = cosine_similarity(qvec, [float(x) for x in vec])
                role_scores[str(it.get("job_id", ""))] = sim
                scored.append((sim, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            filtered = [it for sim, it in scored if sim >= args.min_role_sim]
        else:
            # Fallback: simple word overlap similarity.
            q_tokens = set(re.findall(r"[a-z0-9]+", role_query.casefold()))
            scored2: list[tuple[float, dict[str, Any]]] = []
            for it in filtered:
                r = str(it.get("role", ""))
                r_tokens = set(re.findall(r"[a-z0-9]+", r.casefold()))
                if not q_tokens or not r_tokens:
                    sim = 0.0
                else:
                    sim = len(q_tokens & r_tokens) / len(q_tokens | r_tokens)
                role_scores[str(it.get("job_id", ""))] = sim
                scored2.append((sim, it))
            scored2.sort(key=lambda x: x[0], reverse=True)
            filtered = [it for sim, it in scored2 if sim >= args.min_role_sim]

    # Top-k if requested.
    if args.top_k and args.top_k > 0:
        if role_query:
            filtered = sorted(
                filtered,
                key=lambda it: role_scores.get(str(it.get("job_id", "")), 0.0),
                reverse=True,
            )[: args.top_k]
        else:
            filtered = filtered[: args.top_k]

    out_path = Path(args.out)
    write_json(out_path, {"items": filtered})
    print(f"Filtered to {len(filtered)} postings")
    print(f"Wrote filtered index to {out_path}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    ontology = load_ontology(Path(args.ontology))

    filtered_index = json.loads(Path(args.filtered_index).read_text(encoding="utf-8"))
    items = filtered_index.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Filtered index missing 'items' list")

    postings_dir = Path(args.postings_dir)

    score_acc: dict[str, list[int]] = defaultdict(list)
    parents_counter: dict[str, Counter[str]] = defaultdict(Counter)
    canonical_name: dict[str, Counter[str]] = defaultdict(Counter)

    for it in items:
        job_id = str(it.get("job_id", "")).strip()
        if not job_id:
            continue
        posting_path = postings_dir / f"{job_id}.json"
        if not posting_path.exists():
            # Try the json_path recorded in index.
            jp = it.get("json_path")
            if isinstance(jp, str) and jp:
                posting_path = Path(jp)
        if not posting_path.exists():
            continue

        posting = json.loads(posting_path.read_text(encoding="utf-8"))
        skills = posting.get("skills", [])
        if not isinstance(skills, list):
            continue

        for s in skills:
            if not isinstance(s, dict):
                continue
            name = str(s.get("skill", "")).strip()
            if not name:
                continue
            key = name.casefold()
            canonical_name[key][name] += 1

            try:
                score = int(s.get("score", 0))
            except Exception:
                score = 0
            score = max(0, min(10, score))
            score_acc[key].append(score)

            # Prefer ontology parents if known.
            onto_parents = _case_insensitive_ontology_lookup(ontology, name)
            if onto_parents is not None:
                for p in onto_parents:
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

    aggregated: list[dict[str, Any]] = []
    for key, scores in score_acc.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        avg_score = int(round(avg))
        # Pick most common casing.
        display_name = canonical_name[key].most_common(1)[0][0]
        parents = [p for p, _ in parents_counter[key].most_common()]
        aggregated.append({"skill": display_name, "parents": parents, "score": max(0, min(10, avg_score))})

    aggregated.sort(key=lambda x: (-int(x["score"]), str(x["skill"]).casefold()))

    out_path = Path(args.out)
    write_json(out_path, aggregated)
    print(f"Aggregated {len(aggregated)} skills from {len(items)} postings")
    print(f"Wrote average skills JSON to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract, index, filter, and aggregate job postings.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract metadata + skills per posting, write an index.")
    p_extract.add_argument("--in-dir", default="data/job_postings", help="Directory containing job posting files")
    p_extract.add_argument("--ontology", default="data/skill_ontology.json", help="Skill ontology JSON")
    p_extract.add_argument("--out-dir", default="outputs/job_postings", help="Output directory for per-posting JSON")
    p_extract.add_argument("--index-out", default="outputs/job_postings_index.json", help="Index JSON output")
    p_extract.add_argument("--use-llm", action="store_true", help="Use Gemini via LangChain (requires GOOGLE_API_KEY)")
    p_extract.add_argument("--embed-roles", action="store_true", help="Store Gemini embeddings for role semantic filtering")
    p_extract.set_defaults(func=cmd_extract)

    p_filter = sub.add_parser("filter", help="Filter postings by location (hard) and role (semantic).")
    p_filter.add_argument("--index", default="outputs/job_postings_index.json", help="Job postings index JSON")
    p_filter.add_argument("--location", default="", help="Hard filter: substring match in location")
    p_filter.add_argument("--role-query", default="", help="Semantic role query")
    p_filter.add_argument("--min-role-sim", type=float, default=0.3, help="Minimum role similarity threshold")
    p_filter.add_argument("--top-k", type=int, default=0, help="Keep top K after scoring (0 = no limit)")
    p_filter.add_argument("--out", default="outputs/filtered_job_postings_index.json", help="Filtered index output")
    p_filter.set_defaults(func=cmd_filter)

    p_aggr = sub.add_parser("aggregate", help="Average skills over filtered postings.")
    p_aggr.add_argument("--filtered-index", default="outputs/filtered_job_postings_index.json", help="Filtered index JSON")
    p_aggr.add_argument("--postings-dir", default="outputs/job_postings", help="Directory with per-posting JSON")
    p_aggr.add_argument("--ontology", default="data/skill_ontology.json", help="Skill ontology JSON")
    p_aggr.add_argument("--out", default="outputs/average_job_skills.json", help="Average job skills output JSON")
    p_aggr.set_defaults(func=cmd_aggregate)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
