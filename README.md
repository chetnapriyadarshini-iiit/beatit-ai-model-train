# Beatit-AI — Data Pipeline: Bronze → Silver → Gold (AWS Glue + Redshift)

> **Part of the [Beatit-AI Churn Prediction System](https://github.com/chetnapriyadarshini-iiit)** — a production MLOps system for predicting music streaming subscriber churn on AWS.

This repository contains the AWS Glue ETL jobs and Redshift table definitions that implement a **Bronze → Silver → Gold medallion data architecture** to produce the feature store used by the Beatit churn prediction model.

---

## Table of Contents

- [Overview](#overview)
- [Medallion Architecture](#medallion-architecture)
- [Table Catalogue](#table-catalogue)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Related Repositories](#related-repositories)
- [Contact](#contact)

---

## Overview

Raw user behavioural data from the Beatit music streaming platform is ingested into S3 and progressively refined through three data quality tiers. The final Gold layer (`gold_ml_features`) serves as the feature input to the SageMaker training pipeline, ensuring that the ML model trains on clean, validated, business-aligned features rather than raw event logs.

---

## Medallion Architecture

```
S3 (Raw Data)
     │
     ▼
┌──────────────────────────────────────┐
│  BRONZE  — Raw ingested data         │
│  Minimal transformation, audit trail │
└─────────────────┬────────────────────┘
                  │  AWS Glue ETL
                  ▼
┌──────────────────────────────────────┐
│  SILVER  — Cleaned & validated data  │
│  Deduplication, type casting,        │
│  null handling, schema enforcement   │
└─────────────────┬────────────────────┘
                  │  Feature Engineering
                  ▼
┌──────────────────────────────────────┐
│  GOLD  — Model-ready feature store   │
│  Aggregated, enriched, labelled      │
│  → gold_ml_features (SageMaker input)│
└──────────────────────────────────────┘
```

---

## Table Catalogue

| Table / Folder | Layer | Description |
|---|---|---|
| `dim_user_profile` | Silver | User dimension table — demographics, registration details, subscription tier |
| `fact_subscription` | Silver | Subscription events — plan changes, renewals, cancellations, payment activity |
| `fact_engagement` | Silver | User engagement facts — listening sessions, tracks played, skip rates |
| `daily_aggregates` | Silver | Daily rollup of per-user engagement and subscription metrics |
| `cohort_retention` | Gold | Cohort-level retention rates by registration month and subscription type |
| `revenue_by_cohort` | Gold | Revenue trends segmented by user cohort for churn risk correlation |
| `anomaly_flags` | Gold | Flags for anomalous user behaviour patterns indicative of pre-churn signals |
| `gold_ml_features` | Gold | Final ML feature table — model-ready, labelled, consumed by SageMaker training pipeline |
| `redshift_procedure` | Utility | Stored procedures for incremental table refresh and data quality checks |
| `glue_job_create_tables_for_redshift` | Utility | AWS Glue job scripts to create and populate Redshift tables from S3 |

---

## Repository Structure

```
beatit-ai-glue-redshift-tables/
├── dim_user_profile/               # User dimension table DDL and Glue job
├── fact_subscription/              # Subscription fact table DDL and Glue job
├── fact_engagement/                # Engagement fact table DDL and Glue job
├── daily_aggregates/               # Daily rollup aggregation scripts
├── cohort_retention/               # Cohort retention computation
├── revenue_by_cohort/              # Revenue segmentation by cohort
├── anomaly_flags/                  # Pre-churn anomaly detection flags
├── gold_ml_features/               # Final model-ready feature table
├── redshift_procedure/             # Stored procedures for incremental refresh
└── glue_job_create_tables_for_redshift/  # Glue job entrypoints
```

---

## Technologies Used

| Service / Tool | Purpose |
|---|---|
| **AWS Glue** | Serverless ETL — Bronze → Silver → Gold transformations |
| **Amazon Redshift** | Cloud data warehouse — feature storage and analytical queries |
| **Amazon S3** | Raw data lake — source for Bronze tier ingestion |
| **Python (PySpark)** | ETL transformation logic within Glue jobs |
| **SQL** | Redshift DDL, stored procedures, and aggregation queries |

---

## Related Repositories

| Repository | Role |
|---|---|
| [beatit_ai_common_utilites](https://github.com/chetnapriyadarshini-iiit/beatit_ai_common_utilites) | Shared utilities used across all pipeline components |
| [beatit-ai-model-train](https://github.com/chetnapriyadarshini-iiit/beatit-ai-model-train) | Consumes `gold_ml_features` for SageMaker model training |
| [beatit-ai-model-deploy](https://github.com/chetnapriyadarshini-iiit/beatit-ai-model-deploy) | Deploys trained model to SageMaker endpoints |
| [beatit-ai-model-monitor](https://github.com/chetnapriyadarshini-iiit/beatit-ai-model-monitor) | Monitors deployed endpoints for data and model drift |

---

## Contact

Created by [@chetnapriyadarshini](https://github.com/chetnapriyadarshini) — feel free to reach out with questions or suggestions.
