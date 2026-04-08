from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


# ── Action ────────────────────────────────────────────────────────────────────

class CleaningAction(BaseModel):
    """One data-cleaning operation the agent wants to apply."""

    operation: Literal[
        "fill_nulls",       # fill missing values in a column
        "drop_duplicates",  # remove duplicate rows
        "fix_type",         # cast a column to the correct dtype
        "normalise_column", # standardise inconsistent category labels
        "drop_column",      # remove an irrelevant/corrupted column
        "fix_date_format",  # parse & unify mixed date strings
        "clip_outliers",    # winsorise extreme numeric values
        "rename_column",    # fix a misnamed column header
        "submit",           # agent declares it is done
    ]
    column: Optional[str] = None      # target column (None for row-wise ops)
    params: Dict[str, Any] = {}       # operation-specific parameters
    reasoning: str = ""               # agent explains its action (scored)


# ── Observation ───────────────────────────────────────────────────────────────

class ColumnSummary(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any]          # up to 5 representative values


class CleaningObservation(BaseModel):
    task_id: str                      # "easy" | "medium" | "hard"
    step: int
    done: bool
    reward: Optional[float]

    shape: List[int]                  # [rows, cols]
    columns: List[ColumnSummary]
    issues_detected: List[str]        # human-readable problem list
    data_quality_score: float         # 0.0–1.0, updated every step
    message: str                      # feedback on last action


# ── State ─────────────────────────────────────────────────────────────────────

class CleaningState(BaseModel):
    episode_id: Optional[str] = None
    task_id: str = "easy"
    step_count: int = 0
    max_steps: int = 20
    cumulative_reward: float = 0.0
    actions_taken: List[str] = []
