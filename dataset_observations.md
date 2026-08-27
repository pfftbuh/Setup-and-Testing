# Dataset Observations

## Overview

The dataset contains 97 session-level observations and 81 columns. Each row represents one recorded session. The dataset contains 79 potential predictor variables, a `session_id` identifier, and a binary `label`.

The predictor variables consist of:

- Eight gaze-distribution features: normalized gaze centroid, horizontal and vertical spread, elongation ratio, entropy, peak ratio, and coverage ratio.
- Sixty-four spatial heatmap features representing a grid of screen regions.
- Seven behavioral features: frantic eye-movement violations, forbidden-key violations, off-screen violations, duration violations, number of gaze transitions, percentage of time outside the center region, and overall violation rate.

## Class Distribution

The classes are approximately balanced:

- Label 0: 49 sessions, or 50.5% of the dataset.
- Label 1: 48 sessions, or 49.5% of the dataset.

This balance is useful for binary classification because the model is less likely to achieve an apparently high accuracy simply by predicting the majority class.

## Data Quality

No missing values were observed in the dataset. All predictor columns contain numeric values, while `session_id` is a string identifier and `label` is the target variable. No duplicate session IDs were found in the inspected data.

## Observed Feature Patterns

Sessions assigned to label 1 generally show more active and dispersed gaze behavior than sessions assigned to label 0. The strongest observed differences include:

- A higher number of gaze transitions. The approximate mean difference between label 1 and label 0 was 250 transitions.
- More forbidden-key violations, with an approximate mean difference of 4.9 violations.
- Higher heatmap coverage, indicating that gaze activity is distributed across a larger portion of the screen.
- Higher entropy, indicating a less concentrated gaze distribution.
- A lower peak ratio, indicating that gaze activity is less concentrated in a single dominant region.
- More time spent outside the center region.

These observations suggest that the label-1 sessions are associated with wider screen exploration, more variable gaze behavior, and more recorded behavioral violations. However, these are associations and should not be interpreted as evidence of causation.

## Feature Relationships

Several features are strongly correlated with one another:

- Entropy and coverage ratio have a correlation of approximately 0.94.
- Entropy and peak ratio have a correlation of approximately -0.92.
- Coverage ratio and peak ratio have a correlation of approximately -0.91.

These relationships are expected because all three features describe aspects of the same gaze-distribution pattern. However, the redundancy may reduce the amount of independent information available to the model and can make feature-importance interpretations less reliable.

## Potential Outliers

Some features contain unusually large values relative to the rest of the dataset:

- `elongation_ratio` reaches approximately 26.31, while most observations are substantially lower.
- `num_transitions` reaches approximately 1,281.
- Several violation-count and duration features also contain relatively high values.

These observations should be inspected to determine whether they represent genuine behavior, unusually long sessions, tracking errors, or data-processing artifacts. Outliers should not be removed automatically, since unusual behavior may be relevant to the classification task.

## Suitability as a Prediction-Model Foundation

The dataset is a suitable starting point for developing and testing a prototype prediction model. It has a clear target variable, a nearly balanced class distribution, consistent session-level features, and multiple feature groups that describe complementary aspects of gaze behavior.

The current training approach is also a reasonable baseline. It uses a Random Forest classifier, stratified five-fold cross-validation, probability calibration, and a saved feature-column order for inference.

Nevertheless, the dataset is small relative to the number of predictors. With 97 sessions and 79 predictors, the model may overfit, and cross-validation results may vary considerably depending on the sampled sessions. Therefore, the current dataset should be treated as an exploratory or proof-of-concept dataset rather than sufficient evidence for deployment.

## Validation Risks and Limitations

The following issues should be addressed before drawing strong conclusions from model performance:

1. If several sessions belong to the same participant, random cross-validation may place sessions from the same participant in both the training and validation folds. This can produce overly optimistic performance estimates. Participant-level splitting should be used when participant identifiers are available.
2. Violation-related features may be closely connected to how the labels were assigned. A separate model trained without violation features should be evaluated to determine whether gaze patterns alone provide predictive value.
3. The labels should represent independently defined ground truth. If the label was assigned because a session was intentionally created as a cheating or non-cheating scenario, the result may measure differences between experimental conditions rather than reliably detecting real-world cheating.
4. The dataset should be tested on new participants and new sessions. Performance on unseen data is more informative than performance on the sessions used to construct the feature dataset.
5. Accuracy should not be used as the only evaluation metric. ROC-AUC, precision, recall, F1-score, PR-AUC, confusion matrices, and calibrated probabilities should also be reported.

## Recommended Future Work

The following steps would strengthen the dataset and the resulting model:

- Collect more sessions from a larger and more diverse participant group.
- Record participant identifiers and use participant-level train/test splits.
- Evaluate models with and without violation features.
- Use repeated cross-validation or bootstrap confidence intervals to quantify uncertainty.
- Test the final model on a separate holdout set collected after model development.
- Investigate the extreme elongation and transition values.
- Standardize recording conditions and document camera position, lighting, calibration quality, session duration, and screen configuration.
- Compare the Random Forest baseline with simpler models such as logistic regression and regularized linear classifiers.
- Examine feature importance using permutation importance or model-agnostic explanations rather than relying only on raw tree importance.

## Summary

The dataset provides a promising foundation for an exploratory gaze-behavior classification model. Label 1 is associated with more dispersed gaze activity, more transitions, higher coverage and entropy, lower peak concentration, and more recorded violations. The balanced classes and absence of missing values are positive properties.

However, the limited number of sessions, possible participant dependence, strong feature correlations, outliers, and possible relationship between violation features and labels limit the strength of the conclusions. The dataset is appropriate for developing a baseline and guiding further data collection, but additional validation is required before the model can be described as reliable for real-world decision-making.
