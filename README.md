# job_skill

### Candidate Name: Sanjana Boggaram J

### Scenario Chosen: 2) Skill-Bridge Career Navigator

### Estimated Time Spent: 6.5-7 hours

Pipeline to extract skills from a resume + job postings, compute a job-skill average after filtering, calculate skill gaps, and generate a resource-backed learning plan.

- Design/architecture: [docs/DESIGN.md](docs/DESIGN.md)
- Demo video link: https://youtu.be/_86U5_bmIKc
- Main scripts live in `scripts/`
- A run.sh is also provided.

## Quick start

### 1) Install deps

`pip install -r requirements.txt`

### 2) Configure API keys

Create/update `.env`:

- `OPENROUTER_API_KEY=...`

### 3) Run the pipeline
Run:
- `python run.sh` 
to run the entire pipeline. This will run all the scripts in order.
If you want to run each script seperately then follow:

1. Extract resume skills:
   - `python scripts/extract_resume_skills.py --use-llm`

2. Extract job postings to JSON (if not already):
   - `python scripts/process_job_postings.py extract`

3. Add role embeddings to each job JSON + rebuild combined index:
   - `python scripts/embed_job_postings_and_rebuild_index.py`

4. Filter jobs + generate average job skills:
   - `python scripts/filter_jobs_and_average.py`

5. Generate gap skills JSON:
   - `python scripts/generate_gap_skills.py`

6. Generate gap summary + filtered resources + learning plan:
   - `python scripts/report_generate.py`
  
### AI Disclosure:
- Used ChatGPT for planning. Conversed back and forth, to come up with the requirements and how the architecture should look.
- Used Copilot free version (with model GPT-5.4) for assisting with coding.
- I verified the generated code by manually going through the high level of the code(to understand the flow of the code) and by running the scripts. I also split the project into several parts (which you can see by the number of python scripts) so that i can verify the output at each level. This stratergy helped me a lot, especially in catching a lot of llm output format nuances and a few logical errors.
- examples:
   -A logical error I found was when averaging the skills of filtered job postings. So say I have two filtered job postings, and one has skill:SQL with score 8, and the second one has the same skill with score 6, the averaging function would average these two and give score 7. This was working fine in the LLM generated code. But when theres a scenario where the first job posting has skill:SQL with score 6 and the second job posting does'nt have this skill, the result from the averaging function was 6. That's wrong and ideally it should give score as 3 ((6+0)/2). Therefore I rejected this suggestion.
   - Also Copilot was telling me that many gemini apis were free, but when I tested them out, I figured out they were'nt. This lead to me google searching for free apis to use and finding one that works.
 
### Tradeoffs & Prioritization
   - Adding an agent orchestrator like Langchain would help the project in its increments, but it was'nt required now, so I did not use it.
   - Using Vector databases for job posting data and study resources data would be a future work.
   - Adding more inputs from user along with resume, like leetcode profile, github profile, etc.
   -I would build these features if I had more time.

### Architecture Diagram:
![Architecture](docs/images/skill.jpg)
