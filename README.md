# Project2-Modeling

## Data Cleaning / Transformation

The data provided by the CDC was well prepared, such that we did not have to do
any additional transformations. Some fields came pre-transformed, as documented
here:

| Field     | Raw Values                       | Transformed Values |
| --------- | -------------------------------- | ------------------ |
| Age       | 18 - 24                          | 1                  |
| Age       | 25 - 29                          | 2                  |
| Age       | 30 - 34                          | 3                  |
| Age       | 35 - 39                          | 4                  |
| Age       | 40 - 44                          | 5                  |
| Age       | 45 - 49                          | 6                  |
| Age       | 50 - 54                          | 7                  |
| Age       | 55 - 59                          | 8                  |
| Age       | 60 - 64                          | 9                  |
| Age       | 65 - 69                          | 10                 |
| Age       | 70 - 74                          | 11                 |
| Age       | 75 - 79                          | 12                 |
| Age       | 80+                              | 13                 |
| Education | None / Kindergarten              | 1                  |
| Education | Grades 1 - 8                     | 2                  |
| Education | Grades 9 - 11                    | 3                  |
| Education | Grade 12 or GED                  | 4                  |
| Education | Some college or technical school | 5                  |
| Education | College graduate                 | 6                  |
| Income    | Less than $10,000                | 1                  |
| Income    | $10,000 to less than $15,000     | 2                  |
| Income    | $15,000 to less than $20,000     | 3                  |
| Income    | $20,000 to less than $25,000     | 4                  |
| Income    | $25,000 to less than $35,000     | 5                  |
| Income    | $35,000 to less than $50,000     | 6                  |
| Income    | $50,000 to less than $75,000     | 6                  |
| Income    | $75,000 or more                  | 8                  |

In most tuned models we utilized VIF to drop features, and some form of under-
or over-sampling. Our threshold used for dropping features based on VIF was a
score of 10.

## Results

Scores
| Model               | Tuning                         | Balanced Accuracy | Precision | Recall | F1 (Accuracy / Weighted Avg) |
| ------------------- | ------------------------------ | ----------------- | --------- | ------ | ---------------------------- |
| Logistic Regression | Baseline                       | 0.561             | 0.53      | 0.14   | 0.86 / 0.83                  |
| Logistic Regression | VIF                            | 0.548             | 0.48      | 0.12   | 0.86 / 0.82                  |
| Logistic Regression | Undersampled, VIF              | 0.736             | 0.30      | 0.76   | 0.72 / 0.76                  |
| Logistic Regression | Grid Search                    | 0.735             | 0.30      | 0.76   | 0.72 / 0.76                  |
| KNN                 | Baseline                       | 0.573             | 0.40      | 0.19   | 0.85 / 0.83                  |
| KNN                 | VIF                            |                   | 0.37      | 0.17   | 0.84 / 0.82                  |
| KNN                 | Grid Search                    |                   | 0.45      | 0.08   | 0.86 / 0.81                  |
| KNN                 | Random Search                  | 0.537             |           |        |                              |
| KNN                 | Undersampled                   |                   | 0.29      | 0.50   | 0.77 / 0.79                  |
| KNN                 | Oversampled                    |                   | 0.30      | 0.36   | 0.79 / 0.80                  |
| KNN                 | Cluster Centroid Undersampling |                   | 0.18      | 0.59   | 0.58 / 0.64                  |
| KNN                 | SMOTE Oversampling             |                   | 0.27      | 0.54   | 0.73 / 0.77                  |
| KNN                 | SMOTEENN Oversampling          |                   | 0.25      | 0.69   | 0.67 / 0.72                  |
| Random Forest       | VIF                            |                   | 0.39      | 0.17   | 0.85 / 0.82                  |
| Random Forest       | VIF, Undersampled              |                   | 0.27      | 0.75   | 0.68 / 0.73                  |
| Random Forest       | VIF, Oversampled               |                   | 0.30      | 0.36   | 0.80 / 0.80                  |
| XGBoost             | VIF                            |                   | 0.52      | 0.12   | 0.86 / 0.82                  |
| XGBoost             | VIF, Oversampled               |                   | 0.29      | 0.78   | 0.71 / 0.75                  |
| XGBoost             | VIF, Undersampled              |                   | 0.29      | 0.79   | 0.70 / 0.74                  |
| XGBoost             | VIF, Oversampled, Grid Search  |                   | 0.30      | 0.73   | 0.73 / 0.77                  |
| XGBoost             | VIF, Undersampled, Grid Search |                   | 0.29      | 0.81   | 0.70 / 0.74                  |

## Credits

### Contributors

[Cynthia Estrella](https://github.com/cynstar)\
[Javier Ibarra-sanchez](https://github.com/ibarrajavi)\
[Racquel Jones](https://github.com/RacquelRobinsonJonesATX)\
[Iker Maruri](https://github.com/trapperkreeper)\
[Fu Thong](https://github.com/kibble)

### Data Attribution

[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
