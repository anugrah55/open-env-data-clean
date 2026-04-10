---
title: Data Clean Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🧹 OpenEnv: Data Clean Environment
### The Real-World Benchmarking for Agentic Data Engineering

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-Compatible-green?style=for-the-badge&logo=pytorch)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/anugrah55/data_clean_env)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 🌟 Overview
**Data Clean Env** is a high-fidelity, production-grade [OpenEnv](https://github.com/meta-pytorch/OpenEnv) implementation designed to evaluate and train Reinforcement Learning (RL) agents on the messy, complex reality of **Data Cleaning**. 

Unlike "toy" environments, this project simulates the exact workflow of a data engineer: identifying schema inconsistencies, handling missing values, casting types, and pruning noise from real-world datasets using the power of `pandas`.

---

## 🛠️ Environment Architecture

### 🧠 Action Space
The agent interacts with the environment through atomic, high-level data operations defined in `models.py`:

| Action | Parameters | Description |
| :--- | :--- | :--- |
| `fill_na` | `column_name`, `value` | Replaces missing values with a specific constant. |
| `drop_na` | `column_name` | Removes rows containing missing data in the target column. |
| `drop_column`| `column_name` | Deletes irrelevant or noisy features from the dataset. |
| `rename_column`| `column_name`, `value`| Fixes naming inconsistencies to match target schemas. |
| `change_type` | `column_name`, `value` | Casts columns to `int`, `float`, or `str` for downstream compatibility. |
| `submit` | - | Finalizes the cleaning process and triggers the programmatic grader. |

### 👁️ Observation Space
The agent perceives the state of the data through a detailed schema:
- **`df_schema`**: Real-time dictionary of column data types.
- **`missing_values`**: Current counts of `NaN` values per column.
- **`head`**: A preview of the first 5 rows to identify formatting patterns.
- **`feedback`**: Semantic descriptions of the impact of the last action.

---

## 📈 Task Progression & Grading

Each task is evaluated by a **deterministic programmatic grader** that compares the agent's output against a "Gold Standard" target, producing a score strictly between **(0.0, 1.0)**.

1.  **🟢 Easy (`easy_clean`)**: 
    - **Goal**: Basic imputation.
    - **Challenge**: Fill missing 'age' values.
2.  **🟡 Medium (`medium_clean`)**: 
    - **Goal**: Noise reduction.
    - **Challenge**: Handle missing values across multiple columns and remove "junk" features.
3.  **🔴 Hard (`hard_clean`)**: 
    - **Goal**: Full schema alignment.
    - **Challenge**: Rename columns, perform safe type casting on dirty strings, and handle complex missing value fallbacks.

---

## 🚀 Quick Start

### 🐳 Run with Docker
```bash
# Build the production image
docker build -t openenv_data_clean:latest -f server/Dockerfile .

# Start the environment server
docker run -p 8000:8000 openenv_data_clean:latest
```

### 🧪 Baseline Inference
We provide a deterministic, zero-temperature baseline script using the OpenAI client:
```bash
export HF_TOKEN="your_huggingface_token"
export IMAGE_NAME="openenv_data_clean:latest"
python inference.py
```

---

## ⚖️ Reward Shaping
Our reward function is designed for efficient RL convergence:
- **Incremental Progress**: `+0.1` for every valid schema improvement.
- **Penalization**: `-0.05` for invalid operations (e.g., targetting non-existent columns).
- **Completion Bonus**: A final reward scaling with the total grader score `[0.01 - 0.99]`.

---

## 🎯 Meta Hackathon Compliance
- ✅ **Typed Models**: Fully Pydantic-powered `Observation` and `Action`.
- ✅ **API Standard**: Implements `step()`, `reset()`, and `state()`.
- ✅ **Strict Logs**: Emits `[START]`, `[STEP]`, and `[END]` traces exactly as required.
- ✅ **Robustness**: Handles network timeouts and invalid JSON carefully.

---
Built with ❤️ for the Meta & Hugging Face OpenEnv Hackathon.
