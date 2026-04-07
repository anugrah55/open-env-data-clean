---
title: Data Clean Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Data Clean Environment for OpenEnv

## Overview and Motivation
Data cleaning is one of the most time-consuming real-world tasks for data scientists and analysts.
This OpenEnv simulates a data cleaning scenario where an AI agent must clean a dirty pandas DataFrame.
The agent interacts with the DataFrame using discrete operations (filling NaNs, dropping columns, etc.)
and receives a score based on how perfectly it cleans the data according to the task objective.

## Action Space
The environment expects a `DataCleanAction` which performs one atomic change to the dataframe:
- `fill_na`: Provide `column_name` and `value` to fill NaNs.
- `drop_na`: Provide `column_name` to drop rows with NaNs in that column.
- `drop_column`: Provide `column_name` to drop it.
- `rename_column`: Provide `column_name` and `value` (new name).
- `change_type`: Provide `column_name` and `value` ('int', 'float', 'str').
- `submit`: Commit the final dataframe for grading.

## Observation Space
The environment returns a `DataCleanObservation` detailing the current dataframe state:
- `df_schema`: The dictionary representation of column types.
- `missing_values`: A dictionary representation of NaN counts per column.
- `head`: The first 5 rows in string format.
- `feedback`: Text feedback of the last action.
- `last_error`: Text description of any error encountered.

## Tasks and Difficulty
- **easy_clean (Easy)**: Fill missing values in a single column ('age').
- **medium_clean (Medium)**: Handle multiple missing value types and drop an unnecessary column.
- **hard_clean (Hard)**: Handle missing values, rename columns, and change column data types.

## Setup and Usage
1. Build the Docker image:
   `docker build -t openenv_data_clean:latest -f server/Dockerfile .`
2. Run the server locally:
   `docker run -p 8000:8000 openenv_data_clean:latest`
3. Run inference baseline:
   `export HF_TOKEN="your_token"`
   `export IMAGE_NAME="openenv_data_clean:latest"`
   `python inference.py`

## Baseline Scores
- easy_clean: 1.00
- medium_clean: 1.00
- hard_clean: 1.00
