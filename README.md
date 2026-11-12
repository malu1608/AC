# NBA Playoff Prediction (Machine Learning)

Predicting which teams make the **playoffs** in the next season from historical basketball data, with a full data-science pipeline in Python.

## Pipeline
1. **Data cleaning and integration:** 7 real-world datasets (players, teams, coaches, awards, series results…) cleaned, merged and checked for consistency, handling missing values and removing inactive players
2. **Feature engineering:** team and player performance metrics aggregated per season
3. **Dimensionality reduction:** PCA to reduce correlated features
4. **Sliding-window validation:** models are trained on previous seasons and tested on the next one, so the evaluation never looks into the future
5. **Model comparison:** 5 classifiers compared with ROC AUC and F1, with hyperparameter tuning (GridSearch)
6. **Final predictions** for the target season with every model

## Stack
Python · pandas · NumPy · scikit-learn · Jupyter

## Files
| File | What it contains |
|---|---|
| `data/` | Original datasets |
| `data_cleaning.ipynb` | Cleaning and integration |
| `first_experimental.ipynb` and later notebooks | Feature engineering, sliding window, PCA, models and tuning |

---
Academic project developed at FEUP (Faculty of Engineering, University of Porto), Machine Learning course (AC), Master in Informatics and Computing Engineering.
