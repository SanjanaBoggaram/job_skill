# Design Doc — job_skill

## 1) Goal

Given:
- a candidate resume (PDF)
- a synthetic corpus of job postings (plaintext)
- a skill ontology (skill → parent skills)
- a resources dataset (courses/projects by category)

Produce:
1. Resume skills JSON (skill, parents, score)
2. Job postings JSONs (metadata + skills + role embedding)
3. A filtered subset of job postings based on user preferences (location hard filter + role semantic similarity)
4. An average job-skill profile for the filtered postings
5. A skill-gap JSON (job average minus resume)
6. A learning plan report grounded in the provided resources


## 2) Tech stack

### Language/runtime
- Python 3.10+

### Parsing
- `pypdf` for PDF text extraction (resume)

### AI
- OpenRouter API via OpenAI-compatible SDK (`openai` Python package)
  - Chat model: `nvidia/nemotron-3-nano-30b-a3b` (matches the current `extract_resume_skills.py`)
  - Embedding model (roles): `nvidia/llama-nemotron-embed-vl-1b-v2:free`

### Config
- `.env` loaded via `python-dotenv`
  - `OPENROUTER_API_KEY` is required for LLM calls


## 3) Repository layout

### Inputs (`data/`)
- `data/skill_ontology.json`
  - JSON object: `{ "Skill Name": { "parents": ["Parent1", ...] }, ... }`
  - Used in multiple stages:
    - Resume extraction (LLM prompt grounding + heuristic matching)
    - Job posting skill extraction (heuristic and/or LLM)
    - Skill aggregation and gap computation (parent normalization / taxonomy consistency)

- `data/candidate_resume/`
  - currently contains `Sanjana_Boggaram_Resume.pdf`

- `data/job_postings/`
  - synthetic job postings as **plaintext files without extensions** (e.g., `ABC`, `react`, `IT`)

- `data/study_resources/resources`
  - JSON object mapping category/skill → list of resources
  - each resource is like:
    ```json
    {
      "type": "course" | "project",
      "title": "...",
      "platform": "...",
      "link": "...",
      "level": "beginner" | "intermediate" | "advanced",
      "duration": "...",
      "description": "..."
    }
    ```

### Outputs (`outputs/`)
- `outputs/resume_skills.json`
- `outputs/job_postings/*.json`
- `outputs/job_postings_index.json`
- `outputs/filtered_job_postings_index.json` (or similar, depending on script)
- `outputs/average_job_skills.json`
- `outputs/gap_skills.json`
- `outputs/gap_summary.md`
- `outputs/gap_selected_resources.json`
- `outputs/learning_plan.md`


## 4) Data contracts

### 4.1 Skills JSON (resume/job/average/gap)

A skill item is:
```json
{
  "skill": "AWS",
  "parents": ["Cloud"],
  "score": 6
}
```

Notes:
- `score` is on a 0–10 scale.
- `parents` is a list of skill/category names; we prefer ontology parents where possible.

### 4.2 Job posting JSON (per job)

Stored at `outputs/job_postings/<job_id>.json`:
```json
{
  "job_id": "ABC",
  "company_name": "ABC Tech",
  "role": "Java Full Stack Developer Intern",
  "location": "Kammanahalli, Bengaluru, Karnataka",
  "employment_type": "Internship (on-site)",
  "skills": [ ...skill items... ],
  "role_embedding": [0.0123, -0.0456, ...],
  "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
  "source_path": "data/job_postings/ABC"
}
```

### 4.3 Job postings index

`outputs/job_postings_index.json`:
```json
{
  "items": [ ...job posting objects (same fields as per-job JSON)... ]
}
```

The index is built from the per-job JSONs after embeddings are added.


## 5) Pipeline overview (end-to-end)

### Step A — Resume skill extraction
Script: `scripts/extract_resume_skills.py`

1. Extract text from PDF with `pypdf`.
2. Skill extraction:
   - **Primary path (AI):** OpenRouter chat model produces JSON array of skills.
   - **Fallback path (heuristic):** regex matches against ontology + common terms.
3. Output written to `outputs/resume_skills.json`.

#### Resume AI fallback logic (important)
The resume extractor implements multiple layers of “don’t fail” behavior:

- **Mode switch:**
  - `--use-llm` uses OpenRouter.
  - Without `--use-llm`, a local heuristic extractor runs (no API calls).

#### How the local heuristic extractor works (no API calls)

When `--use-llm` is NOT provided, `scripts/extract_resume_skills.py` runs a deterministic, local extractor designed to be fast and safe (never depends on network/API availability). It uses the ontology and a small list of common technical keywords to produce the same output schema as the LLM.

Inputs:
- Resume text (already extracted from the PDF)
- `data/skill_ontology.json` (skill → parent list)

Algorithm outline:
1. **Ontology term matching (primary):**
  - For every skill name in the ontology, it searches the resume text with a case-insensitive, boundary-aware regex.
  - If a skill is found, it records a hit count (`hits = number of matches`).
  - This yields high-precision matches for taxonomy skills like `Database`, `Backend`, `AI/ML`, etc.

2. **Common-term matching (secondary):**
  - In addition to ontology skills, it searches for common resume tech terms (e.g., `Python`, `JavaScript`, `React`, `Node.js`, `AWS`, `Git`, etc.).
  - These terms may or may not exist as keys in `skill_ontology.json`.

3. **Parent assignment:**
  - If the matched skill exists in `skill_ontology.json`, parents are taken directly from the ontology (preferred).
  - If not in ontology, the extractor applies a small rule-based mapping to assign an approximate parent category (examples):
    - Languages (`Python`, `Java`, `C++`) → `Programming Fundamentals`
    - Web UI (`HTML`, `CSS`, `React`) → `Frontend`
    - Backend terms (`Node.js`, `REST`, `API`) → `Backend`
    - Cloud providers (`AWS`, `GCP`, `Azure`) → `Cloud`
    - Datastores (`MySQL`, `MongoDB`) → `Database`
  - If no reasonable parent can be inferred, it uses `[]`.

4. **Scoring (0–10):**
  - The heuristic score is derived from how often a term appears in the resume.
  - Conceptually: base confidence + bonus per mention, capped at 10.
  - This is intentionally simple; the LLM path is used when you want deeper scoring based on context (projects, depth of usage, etc.).

5. **Sorting + truncation:**
  - Sorts skills by descending hit count (and then alphabetically).
  - Outputs only the top N (keeps the JSON concise and avoids noise).

Why this exists:
- Guarantees you can always produce a skills JSON even without keys / when APIs are down.
- Provides a baseline, repeatable extraction useful for debugging and comparing against LLM output.

- **Strict output instruction:**
  - System message: “You output only JSON.”
  - User prompt includes the ontology JSON + resume text + explicit schema.

- **Robust JSON parsing:**
  - Strip markdown fences (```json ...```)
  - Try `json.loads(raw)`
  - If it fails, fallback regex: extract the first JSON array `[...]` from the raw text and parse that.
  - If still invalid → raise error.

This is the core “AI fallback logic”: always prefer valid JSON, tolerate common model formatting mistakes, and provide a non-AI extraction path.


### Step B — Job postings extraction to JSON
Script: `scripts/process_job_postings.py extract`

- Reads each plaintext file in `data/job_postings/`.
- Extracts job metadata (role/company/location/type) and skills.
- Writes per-job JSON files into `outputs/job_postings/`.

(These JSONs are later augmented with embeddings.)


### Step C — Add role embeddings + rebuild combined index
Script: `scripts/embed_job_postings_and_rebuild_index.py`

- Iterates over every JSON in `outputs/job_postings/`.
- Computes `role_embedding` from the `role` field using OpenRouter embeddings.
- Writes the updated JSON back to disk.
- Rebuilds `outputs/job_postings_index.json` by collecting all per-job JSONs.

Embedding rationale:
- Location filtering is a hard filter.
- Role filtering is semantic via cosine similarity on `role_embedding`.


### Step D — Filter jobs + compute average job skill profile
Script: `scripts/filter_jobs_and_average.py`

1. Prompt user for location:
   - if `none`: no location filter
   - else uses “%like%” behavior: tokens must appear in location string (case-insensitive)

2. Prompt user for role query:
   - embed the query
   - cosine similarity against each job’s `role_embedding`
   - keep top jobs

3. Average skills over the filtered set:
- per skill: average the score across postings where it appears
- output: `outputs/average_job_skills.json`


### Step E — Compute skill gaps
Script: `scripts/generate_gap_skills.py`

Computes `gap_score = avg_job_score - resume_score_estimate` for each skill in `average_job_skills.json`.

Resume score estimate logic:
- If the resume explicitly contains the same skill → use that score.
- Else, if the resume contains **child skills** whose `parents` include the target skill → use the **average** of those child scores.
  - Example: avg jobs require `AI/ML`, resume doesn’t list `AI/ML`, but lists `Machine Learning` and `Agentic AI` with parent `AI/ML`.
  - Resume estimate for `AI/ML` becomes average(child scores).

Writes: `outputs/gap_skills.json`.


### Step F — Resource selection + learning plan generation
Script: `scripts/report_generate.py`

Inputs:
- `outputs/gap_skills.json`
- `data/study_resources/resources`

Flow:
1. Filter to **positive** gaps (score > 0): skills the user should focus on.
2. **LLM 1:** summarize/prioritize gaps → `outputs/gap_summary.md`
3. For each gap skill, collect candidate resources from:
   - resources keyed by the skill name (if exists)
   - resources keyed by each parent category
4. **LLM 2:** select relevant resources for that gap skill.
   - The prompt uses **resource IDs** (`r0`, `r1`, ...) so the model can only select from provided items.
   - Output is strict JSON: `{ "selected_ids": [...] }`
   - Materialize chosen items → `outputs/gap_selected_resources.json`
5. **Final LLM call:** generate a 4-week plan grounded in the selected resources → `outputs/learning_plan.md`


## 6) Configuration (.env)

Minimum:
- `OPENROUTER_API_KEY=...`

Optional overrides:
- `OPENROUTER_MODEL=nvidia/nemotron-3-nano-30b-a3b`


## 7) Operational notes / limitations

- The dataset is synthetic; metadata extraction quality depends on job posting formatting.
- Scores are model/heuristic-derived; they are directional, not ground-truth.
- `outputs/job_postings_index.json` can become large because it embeds the full skill lists + vectors per job.
  - If this grows, the next step would be a lightweight vector DB (FAISS/Chroma) + keeping JSON as source-of-truth.


## 8) Main scripts (cheat sheet)

- Resume skills: `scripts/extract_resume_skills.py`
- Job extraction: `scripts/process_job_postings.py extract`
- Add embeddings + rebuild index: `scripts/embed_job_postings_and_rebuild_index.py`
- Filter + average: `scripts/filter_jobs_and_average.py`
- Gap skills: `scripts/generate_gap_skills.py`
- Plan/report: `scripts/report_generate.py`
