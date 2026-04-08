import asyncio
import os
import textwrap
from typing import List, Optional
import json

from openai import OpenAI

from client import get_client
from models import DataCleanAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "data_clean_env"
MAX_STEPS = 10
TEMPERATURE = 0.7

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent tasked with cleaning a pandas DataFrame.
    You will be given the current DataFrame schema, missing values count per column, and the first 5 rows.
    You must output a JSON string representing exactly one action to take.
    
    Allowed actions:
    {"action_type": "fill_na", "column_name": "col", "value": "0"}
    {"action_type": "drop_na", "column_name": "col"}
    {"action_type": "drop_column", "column_name": "col"}
    {"action_type": "rename_column", "column_name": "old_col", "value": "new_col"}
    {"action_type": "change_type", "column_name": "col", "value": "int"}  (value can be int, float, or str)
    {"action_type": "submit"}
    
    Your goal:
    - easy_clean: Fill missing values in 'age' with '0'.
    - medium_clean: Drop rows with missing values in 'name' and 'age'. Drop column 'ignore_me'.
    - hard_clean: Rename 'EmployeeID' to 'emp_id'. Drop 'Dept' column. Make 'Salary' valid (fill NaN with '0' and convert to float/int). Fill NaN in 'JoinDate' with '2000-01-01'.
    
    When you are done cleaning according to the goal, output {"action_type": "submit"}.
    Reply ONLY with valid JSON.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs_dict: dict) -> dict:
    user_prompt = f"Observation:\n{json.dumps(obs_dict, indent=2)}\nWhat is your next action?"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            stream=False,
        )
        text = completion.choices[0].message.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "fill_na", "column_name": "invalid", "value": "invalid"}

async def run_task(task_name: str, client: OpenAI, env_client) -> None:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        result = await env_client.reset(task=task_name)
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
                
            obs = result.observation
            obs_dict = {
                "schema": obs.df_schema,
                "missing": obs.missing_values,
                "head": obs.head,
                "feedback": obs.feedback,
                "error": obs.last_error
            }
            
            action_dict = get_model_action(client, obs_dict)
            action_str = json.dumps(action_dict)
            action = DataCleanAction(**action_dict)
            
            result = await env_client.step(action)
            reward = result.reward or 0.0
            done = result.done
            error = result.observation.last_error
            
            rewards.append(reward)
            steps_taken = step
            
            if action.action_type == "submit":
                score = reward # grader sets final reward to score
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            
            if done:
                break
                
        success = score >= 0.5
        
    except Exception as e:
        print(f"[DEBUG] Error running task {task_name}: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    image_name = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
    
    task_name_env = os.getenv("DATA_CLEAN_ENV_TASK")
    tasks_to_run = [task_name_env] if task_name_env else ["easy_clean", "medium_clean", "hard_clean"]

    try:
        env_client = await get_client(image_name)
    except Exception as e:
        print(f"[DEBUG] Failed to start env_client: {e}", flush=True)
        for task in tasks_to_run:
            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        for task in tasks_to_run:
            await run_task(task, client, env_client)
    finally:
        try:
            await env_client.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
