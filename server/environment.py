from __future__ import annotations

import uuid
import random
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dataset generators — produce realistic dirty data for each difficulty level
# ---------------------------------------------------------------------------

def _make_easy_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Easy task: fill missing values + correct obvious wrong types.
    ~100 rows, 5 columns, simple nulls and one dtype error.
    """
    random.seed(42)
    np.random.seed(42)
    n = 80

    df = pd.DataFrame({
        "customer_id": range(1001, 1001 + n),
        "age":         [random.randint(18, 75) if random.random() > 0.15 else None for _ in range(n)],
        "salary":      [str(random.randint(30000, 120000)) for _ in range(n)],   # stored as str — wrong type
        "city":        [random.choice(["Mumbai", "Delhi", "Bangalore", None, "Chennai"]) for _ in range(n)],
        "active":      [random.choice([True, False]) for _ in range(n)],
    })

    # Ground truth: nulls filled with median/mode, salary cast to int
    gt = df.copy()
    gt["age"]    = gt["age"].fillna(int(pd.to_numeric(gt["age"]).median()))
    gt["salary"] = gt["salary"].astype(int)
    gt["city"]   = gt["city"].fillna(gt["city"].mode()[0])

    return df, gt


def _make_medium_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Medium task: fuzzy deduplication + normalise inconsistent category labels.
    ~150 rows with deliberate duplicates and messy categories.
    """
    random.seed(7)
    np.random.seed(7)

    categories_dirty = [
        "engineering", "Engineering", "ENGINEERING", "eng",
        "marketing",   "Marketing",   "MARKETING",   "mktg",
        "hr",          "HR",          "human resources",
        "finance",     "Finance",     "FINANCE",
    ]
    canonical = {
        "engineering": "Engineering", "eng": "Engineering", "ENGINEERING": "Engineering",
        "Engineering": "Engineering",
        "marketing": "Marketing", "mktg": "Marketing", "MARKETING": "Marketing",
        "Marketing": "Marketing",
        "hr": "HR", "human resources": "HR", "HR": "HR",
        "finance": "Finance", "FINANCE": "Finance", "Finance": "Finance",
    }

    n = 120
    base = pd.DataFrame({
        "employee_id": range(1, n + 1),
        "name":        [f"Employee_{i}" for i in range(1, n + 1)],
        "department":  [random.choice(list(canonical.keys())) for _ in range(n)],
        "salary":      [random.randint(40000, 150000) for _ in range(n)],
        "join_date":   pd.date_range("2018-01-01", periods=n, freq="3D").strftime("%Y-%m-%d").tolist(),
    })

    # Inject 20 exact duplicates
    dupes = base.sample(20, random_state=1).copy()
    df = pd.concat([base, dupes], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

    # Ground truth
    gt = base.copy()
    gt["department"] = gt["department"].map(canonical)

    return df, gt


def _make_hard_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hard task: mixed locale date formats + schema violations + statistical outliers.
    Agent must fix WITHOUT over-cleaning valid edge cases.
    """
    random.seed(99)
    np.random.seed(99)
    n = 200

    # Mixed date formats (3 locales)
    def rand_date(i):
        d = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 2)
        fmt = i % 3
        if fmt == 0: return d.strftime("%Y-%m-%d")
        if fmt == 1: return d.strftime("%d/%m/%Y")
        return d.strftime("%m-%d-%Y")

    prices  = [round(random.uniform(10, 500), 2) for _ in range(n)]
    # Inject 10 outliers (10× the max) — should be clipped
    for idx in random.sample(range(n), 10):
        prices[idx] = round(random.uniform(5000, 9000), 2)

    # Schema violation: 15 rows have "N/A" string in numeric column
    quantities = [random.randint(1, 100) for _ in range(n)]
    quantity_col = [str(q) if i not in random.sample(range(n), 15) else "N/A" for i, q in enumerate(quantities)]

    df = pd.DataFrame({
        "order_id":    range(10001, 10001 + n),
        "order_date":  [rand_date(i) for i in range(n)],
        "price":       prices,
        "quantity":    quantity_col,            # mixed str/int — schema violation
        "region":      [random.choice(["North", "South", "East", "West"]) for _ in range(n)],
        "discount":    [round(random.uniform(0, 0.4), 2) for _ in range(n)],
    })

    # Ground truth
    gt = df.copy()
    gt["order_date"] = pd.to_datetime(gt["order_date"], format="mixed", dayfirst=False).dt.strftime("%Y-%m-%d")
    gt["quantity"]   = pd.to_numeric(gt["quantity"], errors="coerce").fillna(quantities[0]).astype(int)
    upper = np.percentile([p for p in prices if p < 5000], 95)
    gt["price"]      = gt["price"].clip(upper=upper)

    return df, gt


# ---------------------------------------------------------------------------
# Graders — fully deterministic, return 0.0–1.0
# ---------------------------------------------------------------------------

def _score_easy(current: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    scores = []
    # null elimination
    null_score = 1.0 - (current.isnull().sum().sum() / max(ground_truth.size, 1))
    scores.append(max(0.0, null_score))
    # salary dtype
    dtype_ok = pd.api.types.is_integer_dtype(current.get("salary", pd.Series(dtype=object)))
    scores.append(1.0 if dtype_ok else 0.0)
    # row count preserved
    row_ok = len(current) == len(ground_truth)
    scores.append(1.0 if row_ok else 0.5)
    return round(float(np.mean(scores)), 4)


def _score_medium(current: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    scores = []
    # duplicate removal
    dup_ratio = current.duplicated().sum() / max(len(current), 1)
    scores.append(max(0.0, 1.0 - dup_ratio))
    # department normalisation
    if "department" in current.columns:
        valid_cats = {"Engineering", "Marketing", "HR", "Finance"}
        cat_score = current["department"].isin(valid_cats).mean()
        scores.append(float(cat_score))
    else:
        scores.append(0.0)
    # row count close to ground truth (allow ±5)
    diff = abs(len(current) - len(ground_truth))
    scores.append(1.0 if diff <= 5 else max(0.0, 1.0 - diff / len(ground_truth)))
    return round(float(np.mean(scores)), 4)


def _score_hard(current: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    scores = []

    # date format unified
    if "order_date" in current.columns:
        def is_iso(v):
            try:
                pd.Timestamp(str(v))
                return str(v).count("-") == 2 and len(str(v)) == 10
            except Exception:
                return False
        date_score = current["order_date"].apply(is_iso).mean()
        scores.append(float(date_score))
    else:
        scores.append(0.0)

    # quantity schema fixed (all numeric)
    if "quantity" in current.columns:
        numeric_score = pd.to_numeric(current["quantity"], errors="coerce").notna().mean()
        scores.append(float(numeric_score))
    else:
        scores.append(0.0)

    # outliers clipped (no price > 2× ground_truth max)
    if "price" in current.columns:
        gt_max = ground_truth["price"].max()
        outlier_gone = (current["price"] <= gt_max * 1.1).mean()
        scores.append(float(outlier_gone))
    else:
        scores.append(0.0)

    # no over-cleaning (row count preserved within 5%)
    row_ratio = len(current) / max(len(ground_truth), 1)
    over_clean = 1.0 if 0.95 <= row_ratio <= 1.05 else max(0.0, 1.0 - abs(1.0 - row_ratio))
    scores.append(over_clean)

    return round(float(np.mean(scores)), 4)


# ---------------------------------------------------------------------------
# Issue detector — generates human-readable problem list for observations
# ---------------------------------------------------------------------------

def _detect_issues(df: pd.DataFrame) -> list[str]:
    issues = []
    for col in df.columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            issues.append(f"Column '{col}' has {nulls} missing values")
        if df[col].dtype == object:
            try:
                pd.to_numeric(df[col])
                issues.append(f"Column '{col}' looks numeric but is stored as string")
            except Exception:
                pass
    dups = df.duplicated().sum()
    if dups > 0:
        issues.append(f"{dups} duplicate rows detected")
    return issues if issues else ["No obvious issues detected — consider submitting"]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

TASK_DATASETS = {
    "easy":   _make_easy_dataset,
    "medium": _make_medium_dataset,
    "hard":   _make_hard_dataset,
}

GRADERS = {
    "easy":   _score_easy,
    "medium": _score_medium,
    "hard":   _score_hard,
}

MAX_STEPS = {"easy": 15, "medium": 20, "hard": 25}


class DataCleaningEnvironment:
    def __init__(self):
        self._df: pd.DataFrame = pd.DataFrame()
        self._ground_truth: pd.DataFrame = pd.DataFrame()
        self._task_id: str = "easy"
        self._step: int = 0
        self._episode_id: str = ""
        self._actions_taken: list[str] = []
        self._cumulative_reward: float = 0.0
        self._prev_quality: float = 0.0

    # ── public API ────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> dict:
        assert task_id in TASK_DATASETS, f"Unknown task_id '{task_id}'"
        self._task_id = task_id
        self._step = 0
        self._episode_id = str(uuid.uuid4())
        self._actions_taken = []
        self._cumulative_reward = 0.0

        raw, gt = TASK_DATASETS[task_id]()
        self._df = raw.copy()
        self._ground_truth = gt.copy()
        self._prev_quality = GRADERS[task_id](self._df, self._ground_truth)

        return self._observation(reward=None, done=False, message="Episode started. Inspect the dataset and begin cleaning.")

    def step(self, action: dict) -> dict:
        operation = action.get("operation", "")
        column    = action.get("column")
        params    = action.get("params", {})
        reasoning = action.get("reasoning", "")

        self._step += 1
        self._actions_taken.append(f"{operation}({column})")
        max_steps = MAX_STEPS[self._task_id]

        # Terminal: submit
        if operation == "submit":
            quality = GRADERS[self._task_id](self._df, self._ground_truth)
            reward  = float(quality)
            self._cumulative_reward += reward
            return self._observation(reward=reward, done=True,
                message=f"Submitted. Final quality score: {quality:.3f}")

        # Apply operation
        message, penalty = self._apply(operation, column, params)

        # Score improvement
        quality    = GRADERS[self._task_id](self._df, self._ground_truth)
        improvement = quality - self._prev_quality
        reward     = max(-0.1, improvement) - penalty
        # Small bonus for providing reasoning
        if reasoning.strip():
            reward += 0.01
        reward = round(reward, 4)
        self._prev_quality = quality
        self._cumulative_reward += reward

        done = self._step >= max_steps
        if done:
            message += f" | Max steps reached. Final quality: {quality:.3f}"

        return self._observation(reward=reward, done=done, message=message)

    @property
    def state(self) -> dict:
        return {
            "episode_id":        self._episode_id,
            "task_id":           self._task_id,
            "step_count":        self._step,
            "max_steps":         MAX_STEPS[self._task_id],
            "cumulative_reward": round(self._cumulative_reward, 4),
            "actions_taken":     self._actions_taken,
        }

    # ── operation dispatcher ──────────────────────────────────────────────

    def _apply(self, operation: str, column: str | None, params: dict) -> tuple[str, float]:
        """Returns (message, penalty). Penalty for destructive/wrong actions."""
        df = self._df
        try:
            if operation == "fill_nulls":
                if column not in df.columns:
                    return f"Column '{column}' not found.", 0.05
                strategy = params.get("strategy", "mean")
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                else:
                    fill_val = params.get("value", df[column].mode().iloc[0] if not df[column].mode().empty else "unknown")
                    df[column] = df[column].fillna(fill_val)
                return f"Filled nulls in '{column}' using {strategy}.", 0.0

            elif operation == "drop_duplicates":
                before = len(df)
                df.drop_duplicates(inplace=True)
                df.reset_index(drop=True, inplace=True)
                self._df = df
                return f"Dropped {before - len(df)} duplicate rows.", 0.0

            elif operation == "fix_type":
                if column not in df.columns:
                    return f"Column '{column}' not found.", 0.05
                target = params.get("dtype", "float")
                df[column] = pd.to_numeric(df[column], errors="coerce") if "int" in target or "float" in target else df[column].astype(target)
                if "int" in target:
                    df[column] = df[column].fillna(0).astype(int)
                return f"Cast '{column}' to {target}.", 0.0

            elif operation == "normalise_column":
                if column not in df.columns:
                    return f"Column '{column}' not found.", 0.05
                mapping = params.get("mapping", {})
                if mapping:
                    df[column] = df[column].replace(mapping)
                else:
                    df[column] = df[column].str.strip().str.title()
                return f"Normalised values in '{column}'.", 0.0

            elif operation == "drop_column":
                if column not in df.columns:
                    return f"Column '{column}' not found.", 0.05
                if column in self._ground_truth.columns:
                    return f"Column '{column}' exists in ground truth — dropping it penalises you.", 0.1
                df.drop(columns=[column], inplace=True)
                return f"Dropped column '{column}'.", 0.0

            elif operation == "fix_date_format":
                if column not in df.columns:
                    return f"Column '{column}' not found.", 0.05
                df[column] = pd.to_datetime(df[column], format="mixed", dayfirst=False, errors="coerce").dt.strftime("%Y-%m-%d")
                return f"Unified date format in '{column}' to YYYY-MM-DD.", 0.0

            elif operation == "clip_outliers":
                if column not in df.columns:
                    return f"Column '{column}' not found.", 0.05
                if not pd.api.types.is_numeric_dtype(df[column]):
                    return f"Column '{column}' is not numeric.", 0.05
                lower_pct = params.get("lower_pct", 5)
                upper_pct = params.get("upper_pct", 95)
                lo = np.percentile(df[column].dropna(), lower_pct)
                hi = np.percentile(df[column].dropna(), upper_pct)
                df[column] = df[column].clip(lower=lo, upper=hi)
                return f"Clipped '{column}' to [{lo:.2f}, {hi:.2f}].", 0.0

            elif operation == "rename_column":
                old = column
                new = params.get("new_name")
                if not new:
                    return "rename_column requires params.new_name.", 0.05
                df.rename(columns={old: new}, inplace=True)
                return f"Renamed '{old}' → '{new}'.", 0.0

            else:
                return f"Unknown operation '{operation}'.", 0.05

        except Exception as exc:
            return f"Operation failed: {exc}", 0.05

    # ── observation builder ───────────────────────────────────────────────

    def _observation(self, reward, done, message) -> dict:
        df = self._df
        cols = []
        for c in df.columns:
            sample = df[c].dropna().unique()[:5].tolist()
            cols.append({
                "name":          c,
                "dtype":         str(df[c].dtype),
                "null_count":    int(df[c].isnull().sum()),
                "unique_count":  int(df[c].nunique()),
                "sample_values": [str(v) for v in sample],
            })
        quality = GRADERS[self._task_id](self._df, self._ground_truth)
        return {
            "task_id":            self._task_id,
            "step":               self._step,
            "done":               done,
            "reward":             reward,
            "shape":              list(df.shape),
            "columns":            cols,
            "issues_detected":    _detect_issues(df),
            "data_quality_score": quality,
            "message":            message,
        }
