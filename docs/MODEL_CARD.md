# Model card: Employee attrition classifier

## Summary

This classifier estimates attrition probability from ten employee attributes in the IBM HR Analytics sample dataset. It is built solely as a portfolio and educational demonstration.

## Intended use

- Demonstrating reproducible classification workflows
- Exploring model evaluation under class imbalance
- Showing probability calibration, threshold tuning, and subgroup reporting
- Supporting discussion about the limitations of people analytics

## Out-of-scope uses

Do not use this model to make or support hiring, promotion, compensation, retention, discipline, termination, surveillance, or other employment decisions. Do not use its score as a statement about an individual employee.

## Inputs

Age, monthly income, distance from home, years at the company, job satisfaction, business travel, department, job role, marital status, and overtime.

Age and marital status are demographic or potentially sensitive features. They remain because they were part of the original experiment. Their presence makes this model unsuitable for operational employment use and creates an explicit opportunity to examine subgroup behavior.

## Training and evaluation

- Stratified 80/20 train-test split with a fixed random seed
- Five-fold cross-validated hyperparameter search on the training set
- Candidate models: logistic regression, random forest, gradient boosting
- Primary selection metric: average precision (PR-AUC)
- Sigmoid probability calibration
- Decision threshold selected from out-of-fold training predictions by maximizing F2
- Final evaluation on the untouched test set

Exact generated results are stored in `reports/`.

## Limitations

- The dataset is small and not representative of a specific workforce.
- Historical correlations do not identify causes of attrition.
- Subgroup samples can be too small for stable estimates.
- Calibration and performance may not transfer to another organization or time period.
- The selected threshold reflects a hypothetical preference for recall and has no validated business utility.
- Feature importance does not imply causality or suggest an intervention.

## Fairness and risk

The pipeline reports metrics across age groups, gender, and marital status on the held-out sample. Those slices are diagnostics only. A fairness assessment would also require context-specific harm definitions, confidence intervals, intersectional analysis, affected-worker input, legal review, and external validation.

## Monitoring required for any research extension

Track data drift, missingness, calibration, performance by subgroup, threshold stability, false-positive harms, and feedback loops. Retraining should be governed and documented rather than automatic.
