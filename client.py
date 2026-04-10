from typing import Any, Dict, Optional

from openenv.core.client_types import StepResult
from openenv.core import EnvClient

try:
    from .models import DataCleanAction, DataCleanObservation
    from .server.data_clean_env_environment import DataCleanState
except ImportError:
    from models import DataCleanAction, DataCleanObservation
    from server.data_clean_env_environment import DataCleanState

class DataCleanEnv(
    EnvClient[DataCleanAction, DataCleanObservation, DataCleanState]
):
    def _step_payload(self, action: DataCleanAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DataCleanObservation]:
        obs_data = payload.get("observation", {})
        observation = DataCleanObservation(
            df_schema=obs_data.get("df_schema", ""),
            missing_values=obs_data.get("missing_values", ""),
            head=obs_data.get("head", ""),
            last_error=obs_data.get("last_error"),
            feedback=obs_data.get("feedback"),
            metadata=obs_data.get("metadata", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DataCleanState:
        return DataCleanState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_df_json=payload.get("current_df_json", ""),
            task_name=payload.get("task_name", ""),
            target_df_json=payload.get("target_df_json", ""),
        )

async def get_client(image_name: Optional[str] = None) -> DataCleanEnv:
    if image_name:
        client = await DataCleanEnv.from_docker_image(image_name)
    else:
        client = DataCleanEnv(base_url="http://localhost:8000")
    return client
