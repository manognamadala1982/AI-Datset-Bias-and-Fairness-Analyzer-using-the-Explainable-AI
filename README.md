# Crime Dataset Fairness & Explainable AI Analyzer

A Streamlit learning project for comparing classification models, inspecting
global feature importance, and measuring differences in model outcomes across
selected demographic groups.

## Important limitations

- This is for classroom or retrospective research only. Do not use it to predict
  individual behavior or make policing, charging, sentencing, hiring, or other
  consequential decisions.
- Fairness metrics are descriptive, depend on the selected outcome and groups,
  and do not establish that a model is fair.
- The gender summary counts deduplicated rows in this file, not unique people or
  population-adjusted crime rates. It cannot establish which gender commits
  more crime because the data has no population denominator and may reflect
  reporting, recording, and enforcement patterns.
- Demographic attributes are excluded from model features by design, but other
  fields can still act as proxies.
- Small groups produce unstable rates. Groups below the selected support
  threshold appear in the results but are excluded from fairness gap summaries.
- The application removes exact duplicate rows before splitting, preventing
  identical records from appearing in both train and test data. This does not
  address all possible data leakage or representativeness problems.
- Permutation importance is a global model explanation, not a causal explanation.

## Run locally

From the folder containing `app.py` and `crime_data.csv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads the included CSV by default; another CSV can be uploaded from the
page. By default, it models `Disposition`, evaluates available race/gender
attributes, and compares logistic regression, random forest, and extra trees.
The chosen sensitive attributes are held out of model inputs and used only to
calculate test-set group metrics.

## Metrics

- **Accuracy:** fraction of test predictions that are correct.
- **Balanced accuracy:** average recall across classes, useful with imbalance.
- **Macro F1:** unweighted average of per-class F1 scores.
- **Selection rate:** fraction predicted as the selected positive outcome.
- **Fairness rate (selection-rate parity):** lowest eligible group's selection
  rate divided by the highest group's rate; closer to 100% means those group
  rates are more similar. It is not a universal fairness score or guarantee.
- **True/false positive rate:** class-specific rates calculated for that
  selected outcome.
- **Permutation importance:** mean change in balanced accuracy after shuffling
  one input feature in the holdout set.
