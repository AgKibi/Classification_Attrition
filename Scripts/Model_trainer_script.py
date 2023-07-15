#Data preparation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #scaling features to a normalized range
from sklearn.feature_selection import SelectKBest # Feature selection
from sklearn.decomposition import PCA # Feature selection
from sklearn.pipeline import Pipeline


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning algorithms
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import catboost as ctb
from catboost import CatBoostClassifier

# Model assessment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report # Model assessment
from sklearn.model_selection import GridSearchCV

### Model trainer script

class ModelTrainer:
    def __init__(self):
        self.pipeline = None
        self.grid_search = None

    def build_pipeline(self, classifier_name, feature_selection=None, X_train=None, y_train=None):
        scaler = StandardScaler()
        if feature_selection == 'KBest':
            feature_selector = SelectKBest(k=10)
        elif feature_selection == 'PCA':
            feature_selector = PCA()
        else:
            feature_selector = None

        if classifier_name == 'SVC':
            classifier = SVC()
        elif classifier_name == 'GaussianNB':
            classifier = GaussianNB()
        elif classifier_name == 'LogisticRegression':
            classifier = LogisticRegression()
        elif classifier_name == 'DecisionTree':
            classifier = DecisionTreeClassifier()
        elif classifier_name == 'AdaBoost':
            classifier = AdaBoostClassifier()
        elif classifier_name == 'XGBoost':
            classifier = XGBClassifier()
        elif classifier_name == 'KNN':
            classifier = KNeighborsClassifier()
        elif classifier_name == 'LGBM':
            classifier = LGBMClassifier()
        elif classifier_name == 'CatBoost':
            classifier = CatBoostClassifier(verbose=False) # Disable progress printing (verbose=False)
        else:
            raise ValueError('Invalid classifier name')

        steps = [('scaler', scaler)]
        if feature_selector:
            steps.append(('feature_selector', feature_selector))
        steps.append(('classifier', classifier))

        self.pipeline = Pipeline(steps)

        if X_train is not None and y_train is not None:
            self.pipeline.fit(X_train, y_train)

    def build_grid_search(self, param_grid, cv=5):
        self.grid_search = GridSearchCV(self.pipeline, param_grid, cv=cv)

    def train(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)

    def predict(self, X_test):
        return self.grid_search.predict(X_test)

    def predict_train(self,X_train):
        return self.grid_search.predict(X_train)

    def calculate_metrics(self, target, prediction, average='weighted'):
        accuracy = accuracy_score(target, prediction)
        precision = precision_score(target, prediction, average=average)
        recall = recall_score(target, prediction, average=average)
        f1 = f1_score(target, prediction, average=average)
        mislabeled = (target != prediction).sum()
        total = len(target)
        return accuracy, precision, recall, f1, mislabeled, total

    def print_results(self, metrics, classifier_name=None, X_train=None, y_train=None):
        if classifier_name:
            print(f'{classifier_name} metrics:')
        else:
            print('Metrics:')
        print(f'  Accuracy:  {metrics[0]}')
        print(f'  Precision: {metrics[1]}')
        print(f'  Recall:    {metrics[2]}')
        print(f'  F1 score:  {metrics[3]}')
        print(f'  Mislabeled {metrics[4]} out of {metrics[5]}')
        print('\n')

        # Print selected features for SelectKBest
        if self.pipeline is not None and X_train is not None and y_train is not None:
            feature_selector = None
            for name, step in self.pipeline.named_steps.items():
                if isinstance(step, SelectKBest):
                    feature_selector = step
                    break
        
            if feature_selector is not None:
                feature_selector.fit(X_train, y_train)  # SelectKBest fit on training data
                selected_features = feature_selector.get_support(indices=True)
                feature_names = X_train.columns[selected_features]  # Assuming that X_train is a DataFrame
                print('Selected Features:')
                for feature in feature_names:
                    print(f'  {feature}')
                print('\n')
                print('_____')

