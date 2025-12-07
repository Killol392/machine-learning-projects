# Climate Prediction – Asia (Machine Learning Regression)

This project builds a clean, end-to-end temperature prediction pipeline for Asian cities using **23,221 daily weather records** from the *World Weather Repository* dataset. The goal is to benchmark three regression models-**Ridge, Random Forest, and Gradient Boosting**-after systematic preprocessing, feature engineering, and tuning.

## 1. Project Summary
The dataset contains **41 meteorological, air-quality, and astronomical features**. After filtering for Asia via timezone, the target variable selected was **temperature_celsius**, given its stability and relevance for ML forecasting tasks.

The core objective:
**Predict near-surface temperature (°C) across Asian cities using a robust, reproducible ML pipeline.**

## 2. Data Processing Highlights
- Removed duplicates; handled non-physical negatives via median/mode imputation.
- Capped outliers using **IQR capping** across continuous features to preserve sample size.
- Reduced multicollinearity using **VIF filtering** and domain-based column pruning (e.g., redundant units, timestamps, textual astronomical fields).
- Final feature matrix: **21 cleaned columns → ~63 encoded features** after OneHotEncoding.

## 3. Transformations
- Applied **PowerTransformer (Yeo–Johnson)** for variance stabilization and skew reduction.
- Encoded categorical variables with **OneHotEncoder(min_frequency=0.01)**.
- Validated distributional improvement for top skewed features visually and statistically.

## 4. Feature Selection
- Used both **filter methods** (Correlation + VIF) and **wrapper method (RFECV)**.
- RFECV with Ridge identified **63 optimal encoded features** with CV RMSE ~ **1.89°C**.

## 5. Model Development
Models trained on 80/20 stratified split (temperature-binned):
- **Ridge Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

All wrapped in pipelines to avoid leakage and ensure consistent preprocessing.

## 6. Model Performance (Tuned)
| Model | Test RMSE | R² | Notes |
|-------|-----------|------|-------|
| **Ridge Regression** | ~1.88 | 0.958 | Strong baseline; linear constraints limit extremes. |
| **Random Forest** | ~1.27 | 0.981 | Good non-linear capture; mild overfitting. |
| **Gradient Boosting** | **0.68** | **0.994** | Best overall; excellent generalization and calibration. |

**Best Model: Gradient Boosting Regressor**  
Lowest RMSE, highest R², minimal variance gap between CV and test sets.

## 7. Key Insights
- Temperature patterns are influenced most by **humidity, pressure, wind metrics, and particulate concentrations**.
- Ensemble models outperform linear methods due to non-linear meteorological interactions.
- Power transformation and VIF reduction significantly improved model stability.

## 8. Repository Structure
```
├── Climate Prediction - Asia.ipynb   # Full workflow
├── GlobalWeatherRepository.csv       # Dataset (filtered for Asia in notebook)
└── README.md                         # Project summary
```

## 9. Conclusion
This project demonstrates a complete ML workflow-from raw climate data to tuned prediction models-with Gradient Boosting achieving **~0.68°C RMSE**, making it the most reliable approach for temperature estimation in diverse Asian climates.

