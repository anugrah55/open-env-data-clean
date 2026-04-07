from typing import Literal, Optional, List
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class DataCleanAction(Action):
    """Action for the Data Clean Env environment to manipulate the dataframe."""
    action_type: Literal["fill_na", "drop_na", "rename_column", "drop_column", "change_type", "submit"] = Field(
        ..., description="The type of action to perform."
    )
    column_name: Optional[str] = Field(None, description="The target column name.")
    value: Optional[str] = Field(None, description="The value to use (for fill_na), new name (for rename_column), or new type (for change_type like 'int', 'float', 'str').")

class DataCleanObservation(Observation):
    """Observation from the Data Clean Env environment showing the dataframe state."""
    df_schema: str = Field(default="", description="The schema of the dataframe.")
    missing_values: str = Field(default="", description="A string detailing missing values per column.")
    head: str = Field(default="", description="The first 5 rows of the dataframe.")
    last_error: Optional[str] = Field(default=None, description="Any error from the last action.")
    feedback: Optional[str] = Field(default=None, description="Feedback about the last action.")
