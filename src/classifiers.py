### add classifiers
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

C_LOGISTIC_REGRESSION = [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]
RF_PARAMETERS = {'n_estimators': [100, 200, 20, 50, 100, 200, 500],
             'max_depth': [3, 5, 10],
            'min_samples_leaf': [3, 5]}
MLP_PARAMETERS = {'hidden_layer_sizes':[(100,), (200,), (100, 100), (200, 200)],
             'alpha': [1, 10, 100]}

def fit_classifier(X_train: pd.DataFrame, y_train: pd.DataFrame, model: str, cv=False):
    
    if model=='logistic_regression':
        clf=train_LogisticRegression(X_train, y_train, cv)
        
    elif model=='random_forest':
        clf = train_RandomForest(X_train, y_train, cv)
    elif model == 'mlp':
        clf = train_MLP(X_train, y_train, cv)

    return clf


def drop_zero_cols(X_train, X_test = None, X_test_specific = None):
    all_summed = X_train.sum()
    cols_to_drop = [col for col in X_train.columns if all_summed[col] == 0]
    X_train = X_train.drop(cols_to_drop, axis=1)
    if X_test is None:
        return X_train
    X_test = X_test.drop(cols_to_drop, axis=1)
    if X_test_specific is not None:
        X_test_specific = X_test_specific.drop(cols_to_drop, axis=1)
        #return X_train, X_test, X_test_specific
        return X_train, X_test, X_test_specific   
    return X_train, X_test


def scale_features(X_train, X_test=None, X_test_specific = None):
    X_train_values = X_train.values
    all_means = X_train_values.mean(axis=0)
    all_stds = X_train_values.std(axis=0)
    all_stds[all_stds == 0] = 1

    X_train = pd.DataFrame((X_train_values - all_means) / all_stds, columns=X_train.columns)
    if X_test is None:
        return X_train
    X_test = pd.DataFrame((X_test.values - all_means) / all_stds, columns=X_train.columns)
    if X_test_specific is not None:
        X_test_specific = pd.DataFrame((X_test_specific.values - all_means) / all_stds, columns=X_train.columns)
        return X_train, X_test, X_test_specific
    return X_train, X_test

# def select_features(X_train: pd.DataFrame, y_train: pd.DataFrame, n_features: int = 20000) -> list:
#     '''
#      do RFE to select the top X features
#     '''
#     estimator = LogisticRegression()
#     selector = RFE(estimator, n_features_to_select=n_features, step=0.05)
#     selector = selector.fit(X_train, y_train)
#     selected_feats = selector.get_feature_names_out()

#     return selected_feats

def validate_clf(clf, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_test: pd.DataFrame) -> tuple:
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print('Training accuracy: ', train_acc)
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    print('Training auc: ', train_auc)

    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print('Test accuracy: ', test_acc)
    try:
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print('Test auc: ', test_auc)
    except:
        print('Auc impossible, do you have only one classe?')
        return train_acc, test_acc
    return train_acc, train_auc, test_acc, test_auc

def train_LogisticRegression(X_train: pd.DataFrame, y_train: pd.DataFrame, cv: bool = False) -> LogisticRegression:
    if not cv:
        clf = LogisticRegression(C = 0.001)
        clf.fit(X_train, y_train)
    else:
        clf = LogisticRegressionCV(Cs = C_LOGISTIC_REGRESSION, n_jobs=1, cv=3)
        clf.fit(X_train, y_train)
        print('Best C: ', clf.C_)

    return clf

def train_RandomForest(X_train: pd.DataFrame, y_train: pd.DataFrame, cv: bool = False) -> RandomForestClassifier:
    if not cv:
        clf = RandomForestClassifier(max_depth=10, n_estimators=100)
        clf.fit(X_train, y_train)
    else:
        clf = RandomForestClassifier()
        grid_search = GridSearchCV(clf, RF_PARAMETERS, n_jobs = 1, cv=3)
        grid_search.fit(X_train, y_train)
        print('Best params: ', grid_search.best_params_)
        clf = RandomForestClassifier(**grid_search.best_params_)
        clf.fit(X_train, y_train)

    return clf

def train_MLP(X_train: pd.DataFrame, y_train: pd.DataFrame, cv: bool = False) -> MLPClassifier:
    if not cv:
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), alpha=0.01)
        clf.fit(X_train, y_train)
    else:
        clf = MLPClassifier()
        grid_search = GridSearchCV(clf, MLP_PARAMETERS, n_jobs = 1, cv=3)
        grid_search.fit(X_train, y_train)
        print('Best params: ', grid_search.best_params_)
        clf = MLPClassifier(**grid_search.best_params_)
        clf.fit(X_train, y_train)

    return clf

def fit_classifiers(X_train: pd.DataFrame, y_train: pd.DataFrame, models: list, cv=False):
    trained_models = []
    
    for model in models:
        print(f'Model: {model}')
        if model=='logistic_regression':
            clf=train_LogisticRegression(X_train, y_train, cv)
            
        elif model=='random_forest':
            clf = train_RandomForest(X_train, y_train, cv)
        elif model == 'mlp':
            clf = train_MLP(X_train, y_train, cv)
            
        trained_models.append(clf)
        
        return trained_models
            

def fit_validate_classifiers(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_test: pd.DataFrame, models: list, cv: bool = False) -> tuple:
    trained_models, all_results = [], []
    for model in models:
        print('Model: ', model)
        if model == 'logistic_regression':
            clf = train_LogisticRegression(X_train, y_train, cv)
            results = validate_clf(clf, X_train, y_train, X_test,y_test)
        elif model == 'random_forest':
            clf = train_RandomForest(X_train, y_train, cv)
            results = validate_clf(clf, X_train, y_train, X_test,y_test)
        elif model == 'mlp':
            clf = train_MLP(X_train, y_train, cv)
            results = validate_clf(clf, X_train, y_train, X_test,y_test)
        else:
            print('Not a valid model.')
        print('---')
        trained_models.append(clf)
        all_results.append(results)

    return trained_models, all_results


# def fit_predict_classifiers_converge(X_train: pd.DataFrame, y_train: pd.DataFrame, models: list, cv: bool = False):
#     trained_models = []
    
#     train_preds = []
#     test_preds = []
    
#     for model in models:
#         print(f'Model: {model}')
#         if model =='logistic_regression':
#             clf = train_LogisticRegression(X_train, y_train, cv)
            
#             trained_models.append(clf)
            
#         elif model == 'random_forest':
#             clf = train_RandomForest(X_train, y_train, cv)
            
#             trained_models.append(clf)
            
           
#         elif model == 'mlp':
#             clf = train_MLP(X_train, y_train, cv)
            
#             trained_models.append(clf)
    
#     return trained_models
