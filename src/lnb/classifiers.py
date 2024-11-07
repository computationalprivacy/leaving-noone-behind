### add classifiers
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

C_LOGISTIC_REGRESSION = [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]
RF_PARAMETERS = {
    "n_estimators": [100, 200, 20, 50, 100, 200, 500],
    "max_depth": [3, 5, 10],
    "min_samples_leaf": [3, 5],
}
MLP_PARAMETERS = {
    "hidden_layer_sizes": [(100,), (200,), (100, 100), (200, 200)],
    "alpha": [1, 10, 100],
}


def drop_zero_cols(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
    """
    Drops columns from the input DataFrames where all column values are zero.

    :param X_train: The training data.
    :type X_train: pd.DataFrame
    :param X_test: The test data, optional.
    :type X_test: pd.DataFrame, optional
    :returns: If `X_test` is not provided, returns the modified `X_train` with zero-sum columns dropped.
        If `X_test` is provided, returns both `X_train` and `X_test` with zero-sum columns dropped.
    :rtype: pd.DataFrame or tuple of pd.DataFrame
    """
    all_summed = X_train.sum()
    cols_to_drop = [col for col in X_train.columns if all_summed[col] == 0]
    X_train = X_train.drop(cols_to_drop, axis=1)
    if X_test is None:
        return X_train, None
    X_test = X_test.drop(cols_to_drop, axis=1)
    return X_train, X_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
    """
    Scales the features in X_train by standardizing them.
    If X_test is provided, it scales the test set using the same mean and standard deviation as X_train.

    Parameters:
    X_train: pd.DataFrame
        The training data
    X_test: pd.DataFrame, optional
        The test data (default is None)

    Returns:
    If X_test is not provided, returns the standardized X_train.
    If X_test is provided, returns both the standardized X_train and X_test.
    """

    X_train_values = X_train.values
    all_means = X_train_values.mean(axis=0)
    all_stds = X_train_values.std(axis=0)
    all_stds[all_stds == 0] = 1

    X_train = pd.DataFrame(
        (X_train_values - all_means) / all_stds, columns=X_train.columns
    )
    if X_test is None:
        return X_train, None
    X_test = pd.DataFrame(
        (X_test.values - all_means) / all_stds, columns=X_train.columns
    )
    return X_train, X_test


def validate_clf(
    clf: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple:
    """
    Evaluates the classifier's performance on both the training and test datasets.
    Prints the accuracy and AUC (Area Under the Curve) for both training and test sets.

    Parameters:
    -----------
    clf: ClassifierMixin
        The classifier to be evaluated (any estimator with `predict` and `predict_proba` methods).
    X_train: pd.DataFrame
        The feature data for the training set.
    y_train: pd.DataFrame
        The target labels for the training set.
    X_test: pd.DataFrame
        The feature data for the test set.
    y_test: pd.DataFrame
        pd.DataFrameThe target labels for the test set.

    Returns:
    --------
    tuple: A tuple containing:
      - Training accuracy
      - Training AUC
      - Test accuracy
      - Test AUC (if computable)
    """

    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print("Training accuracy: ", train_acc)
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    print("Training auc: ", train_auc)

    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("Test accuracy: ", test_acc)
    try:
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print("Test auc: ", test_auc)
    except:
        print("Auc impossible, do you have only one class?")
        return train_acc, test_acc
    return train_acc, train_auc, test_acc, test_auc


def train_LogisticRegression(
    X_train: pd.DataFrame, y_train: pd.DataFrame, cv: bool = False
) -> LogisticRegression:
    """
    Trains a logistic regression model using either a fixed regularization parameter or cross-validation.

    Parameters:
    ------------
    X_train: pd.DataFrame
        The feature data for the training set.
    y_train: pd.DataFrame
        The target labels for the training set.
    cv: bool, optional
        If True, performs cross-validation to find the best regularization parameter.
        If False, trains the model using a fixed regularization parameter (default is False).

    Returns:
    ---------
    LogisticRegression: A trained logistic regression model.

    If `cv=True`, the function also prints the best regularization parameter (`C`) found during cross-validation.
    """
    if not cv:
        clf = LogisticRegression(C=0.001)
        clf.fit(X_train, y_train)
    else:
        clf = LogisticRegressionCV(Cs=C_LOGISTIC_REGRESSION, n_jobs=1, cv=3)
        clf.fit(X_train, y_train)
        print("Best C: ", clf.C_)

    return clf


def train_RandomForest(
    X_train: pd.DataFrame, y_train: pd.DataFrame, cv: bool = False
) -> RandomForestClassifier:
    """
    Trains a random forest classifier using either fixed hyperparameters or cross-validation for hyperparameter tuning.

    Parameters:
    -----------
    X_train: pd.DataFrame
        The feature data for the training set.
    y_train: pd.DataFrame
        The target labels for the training set.
    cv: bool, optional
        If True, performs cross-validation to find the best hyperparameters.
                         If False, trains the model using fixed hyperparameters (default is False).

    Returns:
    --------
    RandomForestClassifier: A trained random forest classifier.

    If `cv=True`, the function performs a grid search to tune the hyperparameters and prints the best parameters found.
    """
    if not cv:
        clf = RandomForestClassifier(max_depth=10, n_estimators=100)
        clf.fit(X_train, y_train)
    else:
        clf = RandomForestClassifier()
        grid_search = GridSearchCV(clf, RF_PARAMETERS, n_jobs=1, cv=3)
        grid_search.fit(X_train, y_train)
        print("Best params: ", grid_search.best_params_)
        clf = RandomForestClassifier(**grid_search.best_params_)
        clf.fit(X_train, y_train)

    return clf


def train_MLP(
    X_train: pd.DataFrame, y_train: pd.DataFrame, cv: bool = False
) -> MLPClassifier:
    """
    Trains a Multi-layer Perceptron (MLP) classifier using either fixed hyperparameters or cross-validation for hyperparameter tuning.

    Parameters:
    -----------
    X_train: pd.DataFrame
        The feature data for the training set.
    y_train: pd.DataFrame
        The target labels for the training set.
    cv: bool, optional
        If True, performs cross-validation to find the best hyperparameters.
        If False, trains the model using fixed hyperparameters (default is False).

    Returns:
    --------
    MLPClassifier: A trained MLP classifier.

    If `cv=True`, the function performs a grid search to tune the hyperparameters and prints the best parameters found.
    """

    if not cv:
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), alpha=0.01)
        clf.fit(X_train, y_train)
    else:
        clf = MLPClassifier()
        grid_search = GridSearchCV(clf, MLP_PARAMETERS, n_jobs=1, cv=3)
        grid_search.fit(X_train, y_train)
        print("Best params: ", grid_search.best_params_)
        clf = MLPClassifier(**grid_search.best_params_)
        clf.fit(X_train, y_train)

    return clf


def fit_classifier(
    X_train: pd.DataFrame, y_train: pd.DataFrame, model: str, cv=False
):
    """
    Trains a classifier based on the specified model type using the provided training data.

    Parameters:
    -----------
    X_train: pd.DataFrame
        The feature data for the training set.
    y_train: pd.DataFrame
        The target labels for the training set.
    model: str
        The type of classifier to train. Supported values are: 'logistic_regression', 'random_forest', 'mlp'.
    cv: bool, optional
        If True, performs cross-validation during model training to tune hyperparameters.
        If False, trains the model using fixed hyperparameters (default is False). This is a modification

    Returns:
    --------
    object: A trained classifier object based on the specified model.
    """

    if model == "logistic_regression":
        clf = train_LogisticRegression(X_train, y_train, cv)

    elif model == "random_forest":
        clf = train_RandomForest(X_train, y_train, cv)
    elif model == "mlp":
        clf = train_MLP(X_train, y_train, cv)

    return clf


def fit_classifiers(
    X_train: pd.DataFrame, y_train: pd.DataFrame, models: list, cv=False
):
    """_summary_

    :param X_train: _description_
    :type X_train: pd.DataFrame
    :param y_train: _description_
    :type y_train: pd.DataFrame
    :param models: _description_
    :type models: list
    :param cv: _description_, defaults to False
    :type cv: bool, optional
    :return: _description_
    :rtype: _type_
    """
    trained_models = []

    for model in models:
        print(f"Model: {model}")
        if model == "logistic_regression":
            clf = train_LogisticRegression(X_train, y_train, cv)

        elif model == "random_forest":
            clf = train_RandomForest(X_train, y_train, cv)
        elif model == "mlp":
            clf = train_MLP(X_train, y_train, cv)

        trained_models.append(clf)

    return trained_models


def fit_validate_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    models: list,
    cv: bool = False,
) -> tuple:
    """
    Trains and validates multiple classifiers on the provided training and test sets.

    Parameters:
    ------------
    X_train: pd.DataFrame
        The feature data for the training set.
    y_train: pd.DataFrame
        The target labels for the training set.
    X_test: pd.DataFrame
        The feature data for the test set.
    y_test: pd.DataFrame
        The target labels for the test set.
    models: list
        A list of model names (as strings) to be trained and validated. Supported models are: 'logistic_regression', 'random_forest', 'mlp'.
    cv: bool, optional
        If True, performs cross-validation during model training to tune hyperparameters.
        If False, trains the model using fixed hyperparameters (default is False).

    Returns:
    ---------
    tuple: A tuple containing:
      - A list of trained models.
      - A list of the results (training and test accuracy, AUC, etc.) for each model.
    """
    trained_models, all_results = [], []
    for model in models:
        print("Model: ", model)
        if model == "logistic_regression":
            clf = train_LogisticRegression(X_train, y_train, cv)
            results = validate_clf(clf, X_train, y_train, X_test, y_test)
        elif model == "random_forest":
            clf = train_RandomForest(X_train, y_train, cv)
            results = validate_clf(clf, X_train, y_train, X_test, y_test)
        elif model == "mlp":
            clf = train_MLP(X_train, y_train, cv)
            results = validate_clf(clf, X_train, y_train, X_test, y_test)
        else:
            print("Not a valid model.")
        print("---")
        trained_models.append(clf)
        all_results.append(results)

    return trained_models, all_results
