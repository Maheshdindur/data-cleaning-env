# Data Cleaning Environment

An **OpenEnv-compliant** reinforcement learning environment where AI agents learn
to clean real-world messy tabular datasets, step by step.

## Motivation

Data cleaning is the most time-consuming task in any data engineering or ML workflow —
studies consistently show it accounts for 60–80% of a data scientist's time. Yet
no standard RL environment exists to train or evaluate agents on it. This environment
fills that gap with deterministic, programmatically gradeable tasks that mirror
real cleaning workflows.

---

## Action Space

Each step the agent submits one `CleaningAction`:

| Field | Type | Description |
|---|---|---|
| `operation` | string | One of the operations listed below |
| `column` | string \| null | Target column name (null for row-level ops) |
| `params` | dict | Operation-specific parameters |
| `reasoning` | string | Agent's explanation (small reward bonus) |

### Available operations

| Operation | Description | Key params |
|---|---|---|
| `fill_nulls` | Fill missing values | `strategy`: mean\|median\|mode\|value |
| `drop_duplicates` | Remove duplicate rows | — |
| `fix_type` | Cast column to correct dtype | `dtype`: int\|float\|str |
| `normalise_column` | Standardise category labels | `mapping`: {old: new} |
| `drop_column` | Remove irrelevant column | — |
| `fix_date_format` | Unify mixed date strings to YYYY-MM-DD | — |
| `clip_outliers` | Winsorise extreme values | `lower_pct`, `upper_pct` |
| `rename_column` | Fix wrong column header | `new_name` |
| `submit` | Declare dataset clean — ends episode | — |

---

## Observation Space

Each `step()` and `reset()` returns:

```json
{
  "task_id": "easy",
  "step": 3,
  "done": false,
  "reward": 0.12,
  "shape": [80, 5],
  "columns": [
    {
      "name": "age",
      "dtype": "float64",
      "null_count": 12,
      "unique_count": 45,
      "sample_values": ["23", "41", "None", "67", "29"]
    }
  ],
  "issues_detected": [
    "Column 'age' has 12 missing values",
    "Column 'salary' looks numeric but is stored as string"
  ],
  "data_quality_score": 0.41,
  "message": "Filled nulls in 'age' using median."
}
```

---

## Tasks

### Easy — Fill nulls & fix types
- **Dataset**: 80-row customer table (age, salary, city, active)
- **Problems**: 15% null values in age and city; salary stored as string
- **Grader**: Null elimination score + dtype correctness + row count preservation
- **Max steps**: 15
- **Expected frontier model score**: ~0.85

### Medium — Deduplicate & normalise categories
- **Dataset**: 140-row employee table with 20 injected duplicates; department names in 14 inconsistent variants
- **Problems**: Exact duplicate rows; "engineering", "Engineering", "ENGINEERING", "eng" all mean the same thing
- **Grader**: Duplicate removal rate + category normalisation accuracy + row count proximity to ground truth
- **Max steps**: 20
- **Expected frontier model score**: ~0.70

### Hard — Date formats + schema violations + outliers (without over-cleaning)
- **Dataset**: 200-row orders table
- **Problems**: Three mixed date locale formats; "N/A" strings in numeric quantity column; 10 price outliers (10× normal range)
- **Grader**: Date unification + schema repair + outlier clipping + over-cleaning penalty
- **Max steps**: 25
- **Expected frontier model score**: ~0.50

---

## Reward Function

Rewards are **dense** — provided on every step, not just at episode end.

```
step_reward = (quality_after - quality_before) - penalty + reasoning_bonus
```

- `quality_after - quality_before`: improvement in data quality score (0.0–1.0)
- `penalty`: 0.05 for targeting non-existent columns or unknown operations; 0.10 for dropping ground-truth columns
- `reasoning_bonus`: +0.01 if the agent provides non-empty reasoning
- `submit_reward`: final quality score at time of submission

This ensures the agent receives gradient signal on every action, not just at the end.

---

## Setup & Usage

### Local development

```bash
git clone <your-repo>
cd data_cleaning_env
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --reload --port 8000

# In another terminal — quick sanity check
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}'
```

### Docker

```bash
# From project root
docker build -f server/Dockerfile -t data-cleaning-env .
docker run -p 8000:8000 data-cleaning-env

# Verify
curl http://localhost:8000/health
```

### Python client

```python
from client import DataCleaningEnv

env = DataCleaningEnv(base_url="http://localhost:8000")

# Reset to a task
result = env.reset(task_id="medium")
obs = result["observation"]
print(obs["issues_detected"])

# Take a step
result = env.step({
    "operation": "drop_duplicates",
    "column": None,
    "params": {},
    "reasoning": "Dataset has duplicate rows that need removal"
})
print(result["observation"]["data_quality_score"])
```

---

## Baseline Inference Script

Runs the baseline model against all 3 tasks using the OpenAI API client.

```bash
# With Groq (free tier)
export OPENAI_API_KEY=gsk_your_groq_key
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_MODEL=llama-3.1-8b-instant

python baseline.py
```

### Baseline scores (llama-3.1-8b-instant, 5 episodes each)

| Task | Avg Quality | Avg Reward |
|---|---|---|
| Easy | 0.71 | 0.43 |
| Medium | 0.58 | 0.31 |
| Hard | 0.39 | 0.18 |
| **Overall** | **0.56** | — |

*Scores are reproducible — datasets are seeded (seed=42/7/99 per task).*

---

## HF Spaces Deployment

1. Create a new Space on huggingface.co (Docker SDK)
2. Tag it with `openenv`
3. Push this repository — the Space will build and serve on port 8000

```bash
git remote add space https://huggingface.co/spaces/yourname/data-cleaning-env
git push space main
```

---

## Project Structure

```
data_cleaning_env/
├── models.py              # Pydantic typed contracts (Action, Observation, State)
├── client.py              # Sync + async Python clients
├── baseline.py            # OpenAI API baseline inference script
├── requirements.txt
├── openenv.yaml           # Environment manifest
├── README.md
└── server/
    ├── environment.py     # 3 tasks, graders, all game logic
    ├── app.py             # FastAPI server (HTTP + WebSocket)
    └── Dockerfile
```
