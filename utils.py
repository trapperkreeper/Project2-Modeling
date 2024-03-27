import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from enum import Enum
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from typing import List

def calc_vif(X:pd.DataFrame):
    """
    Calculates the VIF scores for a feature DataFrame.
    """
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i)\
                  for i in range(X.shape[1])]
    vif.sort_values('VIF', ascending=True, inplace=True)
    return(vif)

def confusion_matrix_plot(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cmd = ConfusionMatrixDisplay(cm)
    return cmd

def create_elbow_df(df, num_k):
    inertia = []
    k = list(range(1, num_k + 1))
    for i in k:
        m = KMeans(n_clusters=i, n_init='auto')
        m.fit(df)
        inertia.append(m.inertia_)
    
    elbow_df = pd.DataFrame({
        'k': k,
        'inertia': inertia
    })

    return elbow_df

def create_knn_df(data, max_k):
    k_values = []
    train_scores = []
    test_scores = []
    for k in range(1, max_k, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(data['X_train'], data['y_train'])
        train_scores.append(
            knn.score(data['X_train'], data['y_train'])
        )
        test_scores.append(
            knn.score(data['X_test'], data['y_test'])
        )
        k_values.append(k)
    df = pd.DataFrame({
        'k': k_values,
        'Train Score': train_scores,
        'Test Score': test_scores,
    })
    df['Train - Test'] = df['Train Score'] - df['Test Score']
    return df

# label encoding - better for encoding target vars for classification, rather than features:
def label_encode(col):
    return col.astype("category").cat.codes

def min_max_scaler(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return [X_train_scaled, X_test_scaled]

def one_hot(df, cols, drop=True):
    enc = OneHotEncoder(drop='first', sparse_output=False) if drop\
        else OneHotEncoder(sparse_output=False)
    enc.set_output(transform='pandas')
    encoded = enc.fit_transform(df[cols])
    new_df = df.copy()
    new_df.drop(cols, axis='columns', inplace=True)
    new_df = pd.concat([new_df, encoded], axis='columns')
    return new_df

def standard_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return [X_train_scaled, X_test_scaled]

def vif_removal_priority(X:pd.DataFrame, threshold:int):
    """
    Iteratively drops the feature with highest VIF until all
    scores fall below a given threshold.

    Returns the list of features that should be dropped, paired
    with the calculated VIF.

    :param X: Pandas DataFrame containing the features.
    :param threshold: The VIF threshold over which features should be dropped.
    """
    X_tmp = X.copy()
    drop_list = []
    vif = calc_vif(X_tmp)
    while vif['VIF'].max() >= threshold:
        to_remove = vif.iloc[-1]['variables']
        score = vif.iloc[-1]['VIF']
        drop_list.append([to_remove, score])
        X_tmp.drop(to_remove, inplace=True, axis='columns')
        vif = calc_vif(X_tmp)
    return drop_list

def test_models(models, X_train, X_test, y_train, y_test):
    """
    With the provided models and data, test each combination
    of model and scaling method (Standard Scaler, Min Max Scaler,
    and None).
    """
    results = []
    # X_train_stdscl, X_test_stdscl = standard_scaler(
    #     X_train, X_test
    # )
    # X_train_mmscl, X_test_mmscl = min_max_scaler(
    #     X_train, X_test
    # )

    for name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            'name': name,
            'scaling': 'None',
            'acc_score': accuracy_score(y_test, pred),
            'train_score': model.score(X_train, y_train),
            'score': model.score(X_test, y_test),
            'balanced_score': balanced_accuracy_score(y_test, pred)
        })
        # model.fit(X_train_stdscl, y_train)
        # pred = model.predict(X_test_stdscl)
        # results.append({
        #     'name': name,
        #     'scaling': 'Standard Scaler',
        #     'acc_score': accuracy_score(y_test, pred),
        #     'train_score': model.score(X_train_stdscl, y_train),
        #     'score': model.score(X_test_stdscl, y_test)
        # })
        # model.fit(X_train_mmscl, y_train)
        # pred = model.predict(X_test_mmscl)
        # results.append({
        #     'name': name,
        #     'scaling': 'MinMax Scaler',
        #     'acc_score': accuracy_score(y_test, pred),
        #     'train_score': model.score(X_train_mmscl, y_train),
        #     'score': model.score(X_test_mmscl, y_test)
        # })

    df = pd.DataFrame(results)\
        .sort_values('score', ascending=False)
    return df

Model = Enum('Model', [
    'DECISION_TREE',
    'KNN',
    'LOGISTIC_REGRESSION',
    'RANDOM_FOREST',
    'SVC'
])

def create_model(model_enum: Model, params):
    """
    Currently supports the following params:
     * kernel
     * max_depth
     * max_iter
     * n_estimators
     * n_neighbors
     * random_state
    """
    random_state = params['random_state']\
        if 'random_state' in params else None
    match model_enum:
        case Model.DECISION_TREE:
            return tree.DecisionTreeClassifier(
                random_state=random_state
            )
        case Model.KNN:
            return KNeighborsClassifier(
                n_neighbors=params['n_neighbors']
            )
        case Model.LOGISTIC_REGRESSION:
            C = params['C'] if 'C' in params else 1.0
            if 'max_iter' in params:
                return LogisticRegression(
                    C=C,
                    max_iter=params['max_iter']
                )
            return LogisticRegression(C=C)
        case Model.RANDOM_FOREST:
            n_estimators = params['n_estimators']\
                if 'n_estimators' in params else 100
            max_depth = params['max_depth']\
                if 'max_depth' in params else None
            return RandomForestClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=random_state
            )
        case Model.SVC:
            kernel = params['kernel'] \
                if 'kernel' in params else 'rbf'
            return SVC(kernel=kernel)

def random_search_cv(data, params, model, random_state):
    cv_results = RandomizedSearchCV(model, 
                                    params, 
                                    random_state=random_state, 
                                    verbose=3)
    cv_results.fit(data['X_train'], data['y_train'])
    print(cv_results.best_params_)

    random_pred = cv_results.predict(data['X_test'])
    # Calculate the classification report
    target_names = ["negative", "positive"]
    print(classification_report(data['y_test'],
                                random_pred,
                                target_names=target_names))
