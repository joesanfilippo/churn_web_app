import io
import os
import boto3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
font = {'weight': 'bold'
       ,'size': 16}
plt.rc('font', **font)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

class Churn_Model(object):

    def __init__(self, classifier, split_data):
        """ Initialize an instance of the Churn_Model class given a classifier and number of days to use to 
            classify a user as "churned"
        Args:
            classifier (sklearn model): The type of classifier to use in the model. Examples include Logisitic 
                                        Regression, RandomForestClassifier, and GradientBoostingClassifier.
            split_data (tuple): Training and Test data split into their predictors (X) and targets (y)

        Returns:
            None
            Instantiates a Churn_Model class 
        """
        self.classifier = classifier 
        self.classifier_name = self.classifier.__class__.__name__
        self.X_train, self.X_test, self.y_train, self.y_test = split_data
        
    def convert_cat_to_int(self):
        """ Converts string objects in categorical data to integers to use for training models.
        Args:
            None

        Returns:
            None
            Converts categorical object columns in self.X_train to categorical integer columns.
        """
        full_data = pd.concat([self.X_train, self.X_test], axis=0)
        object_cols = full_data.select_dtypes(include=['object']).columns.tolist()

        for col in object_cols:
            col_dict = {}

            for idx, category in enumerate(full_data[col].unique()):
                col_dict[category] = idx

            self.X_train[col] = self.X_train[col].map(lambda x: col_dict[x])
            self.X_test[col] = self.X_test[col].map(lambda x: col_dict[x])
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train, self.y_train)
        self.X_test = scaler.fit_transform(self.X_test, self.y_train)

    def fit_model(self, grid, selection_type, scoring_type):
        """ Using GridSearchCV or RandomSearchCV, find the optimal hyperparameters of the classifier and 
            then fit it to the training data.
        Args: 
            grid (dict): A dictionary of hyperparameters and their associated values to test the classifier with.
            selection_type (sklearn selection_model): The type of hyperparameter selection to use, either 
                                                      GridSearchCV or RandomizedSearchCV.
            scoring_type (str): The metric used to score the fit, examples include 'accuracy', 'precision', 'recall',
                                or 'roc_auc'.
        Returns:
            None
            Performs a RandomSearchCV or GridsearchCV on a classifier and stores the best_estimator_ to the class.
        """
        self.model_search = selection_type(self.classifier
                                     ,grid
                                     ,n_jobs=-1
                                     ,verbose=False
                                     ,scoring=scoring_type)

        self.model_search.fit(self.X_train, self.y_train)

        print(f"Best Parameters for {self.classifier_name}: {self.model_search.best_params_}")
        print(f"Best {scoring_type} Training Score for {self.classifier_name}: {self.model_search.best_score_:.4f}")

        self.best_model = self.model_search.best_estimator_
        self.y_train_probs = self.best_model.predict_proba(self.X_train)[:,1] 
        self.y_test_probs = self.best_model.predict_proba(self.X_test)[:,1] 

def load_X_y(bucket_name, filename, is_feature_selection=False, feature_list=[]):
    """ Loads the X & y training & test data from AWS Bucket
    Args: 
        bucket_name (str): The AWS S3 bucket to pull the training and test data from.
        is_feature_selection (bool): Whether or not to use specific features to train the model. Default is False.
        feature_list (list): The names of the features to use if `is_feature_selection` is True. Default is an 
                             empty list.

    Returns:
        X_train (Pandas DataFrame): Predictor values for training dataset
        X_test (Pandas DataFrame): Predictor values for test dataset
        y_train (Pandas Series): Target values for training dataset
        y_test (Pandas Series): Target values for test dataset
    """
    aws_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    client = boto3.client('s3'
                          ,aws_access_key_id=aws_id
                          ,aws_secret_access_key=aws_secret)

    train_obj = client.get_object(Bucket=bucket_name, Key=f"{filename}_train.csv")
    test_obj = client.get_object(Bucket=bucket_name, Key=f"{filename}_test.csv")

    X_train = pd.read_csv(io.BytesIO(train_obj['Body'].read()), encoding='utf8')
    X_train = X_train.drop(['user_id', 'signup_time_utc', 'last_order_time_utc'], axis=1)
    
    if is_feature_selection:
        X_train = X_train[feature_list]
    y_train = X_train.pop('churned_user')

    X_test = pd.read_csv(io.BytesIO(test_obj['Body'].read()), encoding='utf8')
    X_test = X_test.drop(['user_id', 'signup_time_utc', 'last_order_time_utc'], axis=1)
    
    if is_feature_selection:
        X_test = X_test[feature_list]
    y_test = X_test.pop('churned_user')

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    
    bucket_name = 'food-delivery-churn'
    filename = 'original_churn'
    is_feature_selection = False
    feature_list = []
        
    split_data = load_X_y(bucket_name, filename, is_feature_selection, feature_list)

    gb_model = Churn_Model(GradientBoostingClassifier(), split_data)
    gradient_boosting_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.25]
                            ,'max_depth': [2, 4, 8]
                            ,'max_features': ['sqrt', 'log2', None]
                            ,'min_samples_leaf': [1, 2, 4]
                            ,'subsample': [0.25, 0.5, 0.75, 1.0]
                            ,'n_estimators': [5,10,25,50,100,200]}
    gb_model.convert_cat_to_int()
    gb_model.fit_model(gradient_boosting_grid, RandomizedSearchCV, 'roc_auc')
    
    print(f"ROC AUC Score on Unseen Data: {roc_auc_score(split_data[3], gb_model.y_test_probs):.3f}")

    best_model = gb_model.best_model
    with open('data/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)