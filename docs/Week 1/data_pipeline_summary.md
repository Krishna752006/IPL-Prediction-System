## Data Pipeline Summary

- Built a modular data pipeline to transform raw IPL match data into model-ready format.  
- Implemented ingestion step to load CSV data and remove invalid rows with missing key fields.  
- Applied One-Hot Encoding to categorical features (teams, venue) for model compatibility.  
- Standardized numerical inputs (`over`, `ball`) and targets (`runs`, `wickets`) using scaling.  
- Combined encoded categorical and scaled numerical features into a single feature matrix (X).  
- Structured target variables as a multi-output regression problem (runs and wickets).  
- Split dataset into train, validation, and test sets using consistent random seeds.  
- Ensured pipeline is fully reproducible via a single execution script (`pipeline.py`).  
- Saved processed dataset in efficient Parquet format for downstream ML usage.  
- Established a clean foundation for training, evaluation, and future MLOps integration.  