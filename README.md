# Driver Behavior ML

Industry-oriented machine learning project for driver behavior analysis using OBD-II time series data.

## Project Goals

This project models:
- Driver intent vs vehicle response
- Latent driving context
- Anomaly detection
- Mechanical stress

The current implementation includes notebook-driven exploration and modeling, and is being upgraded toward a full ML + MLOps workflow.

## Repository Structure

```text
driver-behavior-ml/
|-- data/
|   |-- raw/            # ignored
|   |-- processed/      # ignored
|   `-- features/       # ignored
|-- notebooks/
|   |-- 01_data_cleaning.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_feature_engineering.ipynb
|   |-- 04_context_discovery.ipynb
|   |-- 05_modeling.ipynb
|   `-- 06_visualization.ipynb
|-- src/
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- evaluation/
|   `-- utils/
|-- configs/
|   `-- config.yaml
|-- models/             # ignored
|-- reports/            # ignored
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Data & Outputs

- Dataset is **not included** in the repository.
- All processed data, engineered features, context-enhanced data, and generated outputs are excluded from Git tracking.
- These artifacts will be managed using **DVC (Data Version Control)** in a future phase.
- To run the project, place the dataset inside `data/raw/`.

## Notebooks

All notebooks use project-relative paths and are stored in `notebooks/`:
- `01_data_cleaning.ipynb`
- `02_eda.ipynb`
- `03_feature_engineering.ipynb`
- `04_context_discovery.ipynb`
- `05_modeling.ipynb`
- `06_visualization.ipynb`

Notebook outputs are intentionally cleared for reproducibility and clean version control.

## Setup

```bash
pip install -r requirements.txt
```

## MLOps Roadmap (Next)

- Modularize notebook logic into `src/`
- Add data/version tracking with DVC
- Add experiment tracking and model registry
- Add CI checks and training/evaluation pipelines
