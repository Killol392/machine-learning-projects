# Arthritis Profiling

## Overview
A compact end-to-end analysis for predicting arthritis severity using clinical, biochemical, and lifestyle indicators. The project focuses on building a clean, statistically sound pipeline-from data inspection to feature engineering-ensuring the dataset is fully prepared for downstream machine-learning modeling.

## Dataset
**File:** `APDDataset.xlsx`  
**Samples:** 102  
**Features:** 25 (post-cleaning)  
**Label:** Binary arthritis severity (0/1)

## What This Project Does
- Inspects feature types and data structure
- Cleans missing values using distribution-aware imputation
- Detects outliers via boxplots, Z-score, and variance analysis
- Standardizes all numerical features for model readiness
- Handles class imbalance using undersampling and SMOTE
- Performs targeted EDA to understand clinical patterns
- Removes redundant features with correlation filtering
- Identifies informative features using Mutual Information
- Engineers additional medically relevant features

## Key Insights
- Strong right-skew observed in inflammatory markers (CRP, ESRh, ESRo)
- Minimal relationship between Age and Calcium; no gender-based RBC difference
- High-variance features (TC, ASO, RBS) indicate biological spread rather than noise
- Correlated hematology features (MCV, MCH, PCV, MCHC) removed to reduce redundancy
- Final feature set preserves discriminatory information while reducing multicollinearity

## Final Feature Set
`['Gender_M', 'ESRh', 'Hb', 'RBC', 'Abs', 'PC', 'ASO', 'CRP', 'RBS', 'Urea', 'Creatinine', 'Calcium', 'Uric_Acid']`

## Engineered Features
- **RBC_Hb_Ratio** - hematologic efficiency  
- **Inflammation_Load** - CRP × ESRh  
- **High_CRP_Flag** - binary high-inflammation indicator  

## Usage
1. Install dependencies  
   ```bash
   pip install pandas numpy seaborn matplotlib scipy scikit-learn imbalanced-learn
2. Place APDDataset.xlsx in the project directory
3. Run the notebook: "Arthritis Profile.ipynb"

## Summary
This project delivers a disciplined data-preparation workflow suitable for medical-grade tabular prediction tasks. It emphasizes correctness, reproducibility, and interpretability-resulting in a clean, model-ready dataset and a feature space optimized for downstream classification.
