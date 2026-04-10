from fastapi.testclient import TestClient

from models import DataCleanAction
from server.app import app
from server.data_clean_env_environment import DataCleanEnvironment


def test_easy_clean_solution_scores_expected_value() -> None:
    env = DataCleanEnvironment()
    env.reset(task="easy_clean")
    env.step(DataCleanAction(action_type="fill_na", column_name="age", value="0"))

    result = env.step(DataCleanAction(action_type="submit"))

    assert result.done is True
    assert result.reward == 0.99


def test_medium_clean_wrong_solution_is_not_near_perfect() -> None:
    env = DataCleanEnvironment()
    env.reset(task="medium_clean")
    env.step(DataCleanAction(action_type="fill_na", column_name="age", value="0"))
    env.step(DataCleanAction(action_type="drop_na", column_name="name"))
    env.step(DataCleanAction(action_type="drop_column", column_name="ignore_me"))

    result = env.step(DataCleanAction(action_type="submit"))

    assert result.reward < 0.99


def test_hard_clean_wrong_join_date_is_not_near_perfect() -> None:
    env = DataCleanEnvironment()
    env.reset(task="hard_clean")
    env.step(DataCleanAction(action_type="rename_column", column_name="EmployeeID", value="emp_id"))
    env.step(DataCleanAction(action_type="drop_column", column_name="Dept"))
    env.step(DataCleanAction(action_type="fill_na", column_name="Salary", value="0"))
    env.step(DataCleanAction(action_type="change_type", column_name="Salary", value="float"))
    env.step(DataCleanAction(action_type="fill_na", column_name="JoinDate", value="wrong-date"))

    result = env.step(DataCleanAction(action_type="submit"))

    assert result.reward < 0.99


def test_state_endpoint_keeps_core_state_fields() -> None:
    client = TestClient(app)
    client.post("/reset", json={"task": "easy_clean"})

    response = client.get("/state")

    assert response.status_code == 200
    payload = response.json()
    assert "episode_id" in payload
    assert "step_count" in payload
