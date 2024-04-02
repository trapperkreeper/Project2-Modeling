# Project2-Modeling

## Results

Note: VIF Threshold = 10

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
