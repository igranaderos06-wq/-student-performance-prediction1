# Student Performance Prediction System

## Problem
Academic risk is often noticed only after poor results. This classroom prototype explores whether historical student indicators can flag learners who may need additional academic support.

## Dataset
UCI Machine Learning Repository - Student Performance (Cortez & Silva, 2008), **Mathematics file** (`student-mat.csv`, 395 records).
https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction target
`G3 < 10` = **AT RISK**; `G3 >= 10` = **NOT AT RISK**. `G3` is the target only and is never used as an input feature.

## Technologies
Python, pandas, scikit-learn (Random Forest), joblib, Streamlit, GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains, creates student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
In the app, download the sample input CSV, upload it, review predictions, and download the results.

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to other institutions or populations.
- Small dataset; accuracy alone is not enough - check recall and precision for the "At risk" class.
- G1 and G2 are strong predictors but are only available late in the term.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
