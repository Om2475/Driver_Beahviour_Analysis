# Driver Behaviour Analysis Using OBD-II Time-Series Data

This repository presents an end-to-end applied research pipeline for analyzing driving behaviour from OBD-II telemetry. The workflow spans raw trip logs, signal cleaning, feature engineering, latent context discovery, sequence representation learning, risk modelling, and final interpretability outputs.

The current implementation is notebook-first, with a clear path toward a production MLOps stack.

## 1. Project Summary

The objective is to convert high-frequency vehicle telemetry into interpretable safety intelligence at two levels:

1. Window level: detect short-term risky behaviour patterns.
2. Trip and driver level: aggregate temporal evidence into stable behaviour profiles.

The core modelling idea is to treat risk as a composite phenomenon rather than a single score spike. Risk is analyzed jointly through uncertainty, anomaly evidence, behavioural instability, and temporal regime transitions.

## 2. Research Questions

This project is structured around the following questions:

1. Can latent sequence embeddings separate driving styles in a meaningful way?
2. Which feature families are most associated with elevated risk?
3. How often does high predicted risk coincide with low model confidence?
4. Can we summarize each trip into a stable behaviour archetype useful for downstream action?

## 3. Latest Quantitative Snapshot

The numbers below come from the latest generated analysis artifact at reports/insights_summary.json.

| Metric | Value |
|---|---:|
| Windows analyzed | 78,380 |
| Trips analyzed | 81 |
| Sequence embedding dimension | 128 |
| Trip embedding dimension | 384 |
| Mean risk score | 0.2542 |
| Risk 95th percentile | 0.5539 |
| Risk 99th percentile | 0.6700 |
| Final embedding quality score | 0.7776 |
| Top causal feature (latest run) | throttle_mean_scaled |

Confidence-zone distribution from the same run:

1. uncertain: 89.66%
2. confident_safe: 8.86%
3. confident_risky: 1.49%

## 4. End-to-End Pipeline

The project follows a six-stage notebook pipeline:

1. 01_data_cleaning.ipynb
2. 02_eda.ipynb
3. 03_feature_engineering.ipynb
4. 04_context_discovery.ipynb
5. 05_modeling.ipynb
6. 06_visualization.ipynb

High-level flow:

~~~text
Raw OBD-II logs
	-> cleaning and harmonization
	-> windowed feature construction
	-> context enrichment
	-> sequence embeddings and risk signals
	-> trip/driver aggregation
	-> explainable analytics and report exports
~~~

## 5. Data and Feature Design

### 5.1 Signal Source

The raw inputs are OBD-II trip logs stored under data/raw/OBD-II-Dataset. The file naming pattern indicates trip sessions and context tags.

### 5.2 Configured Windowing

From configs/config.yaml:

1. sampling_rate_hz: 10
2. window_seconds: 10
3. stride_seconds: 5
4. feature schema: v2

This yields overlapping temporal windows that retain local dynamics while supporting robust trip-level aggregation.

### 5.3 Feature Families

The engineered representation combines:

1. Kinematic statistics from speed and acceleration derivatives.
2. Powertrain and control interactions (for example throttle-RPM covariance terms).
3. Temporal consistency and transition indicators.
4. Context-aware and confidence-aware derived signals.

## 6. Modelling and Interpretation Strategy

The modelling stage is designed as a multi-signal risk framework.

1. Sequence encoder: learns compact latent trajectory representations.
2. Behaviour decomposition: separates anomaly, instability, uncertainty, temporal, and behaviour components.
3. Composite risk scoring: integrates the component signals into a unified risk index.
4. Archetype assignment: maps trips into interpretable behaviour classes such as mixed, aggressive, inconsistent, stable, and anomalous.
5. Post-hoc analysis: correlational and causal-style ranking layers identify likely risk drivers.

## 7. Key Findings from the Latest Run

1. Risk is not dominated by one factor; it co-varies with uncertainty and anomaly evidence.
2. Embedding-space structure is usable, with moderate separability and meaningful trip-level organization.
3. A substantial portion of high-risk windows lies in low-confidence regions, reinforcing the need for confidence-aware operational policy.
4. Throttle-related features repeatedly appear among top risk-associated signals in the current configuration.

## 8. Repository Layout

~~~text
driver-behavior-ml/
|-- configs/
|   `-- config.yaml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- features/
|-- notebooks/
|   |-- 01_data_cleaning.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_feature_engineering.ipynb
|   |-- 04_context_discovery.ipynb
|   |-- 05_modeling.ipynb
|   `-- 06_visualization.ipynb
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- features/
|   |-- models/
|   `-- utils/
|-- models/
|-- reports/
|-- requirements.txt
|-- .gitignore
`-- README.md
~~~

## 9. Reproducibility Guide

### 9.1 Environment Setup

~~~bash
pip install -r requirements.txt
~~~

### 9.2 Data Placement

Place raw source files under data/raw before running notebooks.

### 9.3 Execution Order

Run notebooks in numeric order from 01 through 06 to preserve artifact dependencies.

### 9.4 Core Local Artifacts (Generated)

The following outputs are produced locally by the pipeline:

1. data/features/final_driver_behavior.csv
2. data/features/driver_profile.csv
3. data/features/embedding_metrics.csv
4. data/features/feature_importance.csv
5. reports/final_driver_analysis.csv
6. reports/driver_leaderboard.csv
7. reports/trip_risk_summary.csv
8. reports/insights_summary.json

Note: the repository is configured to keep most large data artifacts and generated reports out of version control.

## 10. Engineering Notes

1. Current implementation is notebook-centric by design for rapid research iteration.
2. The src package skeleton is prepared for staged migration into modular Python pipelines.
3. Existing configuration supports deterministic experiment settings via a fixed random seed.

## 11. Limitations

1. Dataset scope is limited to available captured sessions and environments.
2. Ground-truth behavioural labels are partly inferred from derived context layers.
3. Reported metrics are configuration-dependent and should be revalidated after feature or model changes.

## 12. Roadmap

1. Move notebook logic into tested src modules.
2. Introduce strict experiment tracking and model registry.
3. Add CI validation for data contracts and model outputs.
4. Add DVC-backed artifact versioning for reproducible research snapshots.

## 13. References

1. GitHub repository: https://github.com/Om2475/Driver_Beahviour_Analysis
2. OBD background and standardization context: https://en.wikipedia.org/wiki/On-board_diagnostics
3. CARB OBD-II fact sheet: https://ww2.arb.ca.gov/resources/fact-sheets/board-diagnostic-ii-obd-ii-systems-fact-sheet

## 14. Citation

If you use this repository in an academic or industrial report, cite it as:

~~~text
Om2475. Driver Behaviour Analysis Using OBD-II Time-Series Data.
GitHub repository: https://github.com/Om2475/Driver_Beahviour_Analysis
Accessed: 2026-04-20
~~~
