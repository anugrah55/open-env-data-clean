import json
from uuid import uuid4
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import DataCleanAction, DataCleanObservation

class DataCleanState(State):
    current_df_json: str
    task_name: str
    target_df_json: str

class DataCleanEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = DataCleanState(episode_id=str(uuid4()), step_count=0, current_df_json="", task_name="", target_df_json="")
        self._df: pd.DataFrame = pd.DataFrame()
        self._target_df: pd.DataFrame = pd.DataFrame()
        
    def _get_obs(self, feedback: Optional[str] = None, error: Optional[str] = None, done: bool = False, reward: float = 0.0) -> DataCleanObservation:
        schema = str(self._df.dtypes.to_dict())
        missing = str(self._df.isna().sum().to_dict())
        head = self._df.head().to_string()
        return DataCleanObservation(
            df_schema=schema,
            missing_values=missing,
            head=head,
            last_error=error,
            feedback=feedback,
            done=done,
            reward=reward,
        )

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task: str = "easy_clean", **kwargs: Any) -> DataCleanObservation:
        self._state = DataCleanState(episode_id=str(uuid4()), step_count=0, current_df_json="", task_name=task, target_df_json="")
        
        if task == "easy_clean":
            self._df = pd.DataFrame({"id": [1, 2, 3], "age": [25.0, np.nan, 30.0]})
            self._target_df = pd.DataFrame({"id": [1, 2, 3], "age": [25.0, 0.0, 30.0]})
        elif task == "medium_clean":
            self._df = pd.DataFrame({
                "name": ["Alice", "Bob", "Charlie", None], 
                "age": [25.0, np.nan, 30.0, 22.0], 
                "ignore_me": [1, 2, 3, 4]
            })
            self._target_df = pd.DataFrame({
                "name": ["Alice", "Bob", "Charlie"], 
                "age": [25.0, np.nan, 30.0], 
            }).dropna(subset=["name", "age"])
            self._target_df = self._target_df.reset_index(drop=True)
        elif task == "hard_clean":
            self._df = pd.DataFrame({
                "EmployeeID": ["E1", "E2", "E3"],
                "Dept": ["IT", "HR", "IT"],
                "Salary": ["5000", np.nan, "6000"],
                "JoinDate": [np.nan, "2020-01-01", "2021-01-01"]
            })
            self._target_df = pd.DataFrame({
                "emp_id": ["E1", "E2", "E3"],
                "Salary": [5000.0, 0.0, 6000.0],
                "JoinDate": ["2000-01-01", "2020-01-01", "2021-01-01"]
            })
        else:
            self._df = pd.DataFrame({"col": [1, 2]})
            self._target_df = pd.DataFrame({"col": [1, 2]})
            
        self._state.current_df_json = self._df.to_json()
        self._state.target_df_json = self._target_df.to_json()
        
        return self._get_obs(feedback=f"Started task {task}.")

    def step(self, action: DataCleanAction) -> DataCleanObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = 0.0
        error = None
        feedback = None
        done = False
        
        if action.action_type == "submit":
            done = True
            score = self._grade()
            reward = score  # Final reward based on grader
            feedback = f"Submitted. Final score: {score}"
            return self._get_obs(feedback=feedback, done=done, reward=reward)
            
        col = action.column_name
        val = action.value
        
        try:
            if col and col not in self._df.columns:
                raise ValueError(f"Column '{col}' not found.")
                
            if action.action_type == "fill_na":
                if not col or val is None: raise ValueError("fill_na requires column_name and value.")
                # Basic inference of type
                try:
                    typed_val = float(val) if '.' in val else int(val)
                except ValueError:
                    typed_val = val
                self._df[col] = self._df[col].fillna(typed_val)
                feedback = f"Filled NaNs in {col} with {val}."
                reward = 0.1
                
            elif action.action_type == "drop_na":
                if not col: raise ValueError("drop_na requires column_name.")
                self._df = self._df.dropna(subset=[col])
                self._df = self._df.reset_index(drop=True)
                feedback = f"Dropped rows with NaNs in {col}."
                reward = 0.1
                
            elif action.action_type == "drop_column":
                if not col: raise ValueError("drop_column requires column_name.")
                self._df = self._df.drop(columns=[col])
                feedback = f"Dropped column {col}."
                reward = 0.1
                
            elif action.action_type == "rename_column":
                if not col or not val: raise ValueError("rename_column requires column_name and value.")
                self._df = self._df.rename(columns={col: val})
                feedback = f"Renamed column {col} to {val}."
                reward = 0.1
                
            elif action.action_type == "change_type":
                if not col or not val: raise ValueError("change_type requires column_name and value.")
                if val == "int": self._df[col] = self._df[col].astype(int)
                elif val == "float": self._df[col] = self._df[col].astype(float)
                elif val == "str": self._df[col] = self._df[col].astype(str)
                else: raise ValueError("Type must be 'int', 'float', or 'str'.")
                feedback = f"Changed type of {col} to {val}."
                reward = 0.1
                
        except Exception as e:
            error = str(e)
            reward = -0.05
            
        self._state.current_df_json = self._df.to_json()
        return self._get_obs(feedback=feedback, error=error, done=done, reward=reward)

    def _grade(self) -> float:
        task = self._state.task_name
        score = 0.0
        
        if task == "easy_clean":
            if "age" in self._df.columns and self._df["age"].isna().sum() == 0:
                try:
                    if len(self._df["age"]) == len(self._target_df["age"]) and (self._df["age"] == self._target_df["age"]).all():
                        score = 1.0
                except Exception:
                    pass
        
        elif task == "medium_clean":
            max_score = 3.0
            current_score = 0.0
            if "name" in self._df.columns and self._df["name"].isna().sum() == 0:
                current_score += 1.0
            if "age" in self._df.columns and self._df["age"].isna().sum() == 0:
                current_score += 1.0
            if "ignore_me" not in self._df.columns:
                current_score += 1.0
            score = current_score / max_score
            
        elif task == "hard_clean":
            max_score = 4.0
            current_score = 0.0
            if "emp_id" in self._df.columns:
                current_score += 1.0
            if "Dept" not in self._df.columns:
                current_score += 1.0
            if "Salary" in self._df.columns and self._df["Salary"].isna().sum() == 0 and pd.api.types.is_numeric_dtype(self._df["Salary"]):
                current_score += 1.0
            if "JoinDate" in self._df.columns and self._df["JoinDate"].isna().sum() == 0:
                current_score += 1.0
            score = current_score / max_score
            
        return max(0.01, min(0.99, float(score)))

    @property
    def state(self) -> State:
        return self._state
