# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:12:31 2024

@author: arsen
"""

# First, we import the needed libraries.
from joblib import dump
import pandas as pd
from sklearn import set_config
import boto3
from io import BytesIO


# Then, we also set configuration to transform any non-pandas output in the pipe steps
# to the pandas DataFrame.
set_config(transform_output="pandas")

# After that, we load the training data from S3 bucket.
train_data = pd.read_csv('s3://final-project-team-3a/training_data/training_data.csv')

# We get rid of columns that are not relevant for building the model.
train_data = train_data.drop(columns = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])

# We prepare the feature Matrix and the lable array for the training data.
# In the feature matrix we drop the target column 'Attrition_Flag' and in the 
# label array we only keep the target column.
    
X_train, y_train = train_data.drop(['Attrition_Flag'], axis=1), train_data['Attrition_Flag']

# We encode the target column 'Attrition_Flag' to a binary value.
# Existing Customer: 0 --> Existing Customer stands for people that will not 
# churn but will stay with the credit card company.
# Attrited Customer: 1 --> Attrited Customer stands for people that will churn 
# and leave the credit card company.
# We set the minority class (Attrited Customer), the class we are intrested in 
# predicting, to 1. We want to predict the customers that are likely to churn.
# This is crucial given that some algorithms can only handle numeric values -->
# So, we have to represent also the target category as numeric value.
y_train = y_train.map({'Existing Customer': 0, 'Attrited Customer': 1})

# We separate the numerical features from the categorical ones, and store the 
# names of the respective columns in the lists below.
# In that way, we can apply different preprocessing steps to the numerical and 
# categorical features in the next steps.
num_features = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

# Before actual preprocessing, we define the transformer that is going to 
# transform the output of some pipeline steps from numerical arrays to pandas 
# dataframe. Even though, set_config function in the beginning has the same
# purpose it does not always work as expected.
from sklearn.base import BaseEstimator, TransformerMixin

class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.column_names)

# Then, we import and define the elements needed for both column transformers.

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# We specify two rather standard preprocessing pipelines, one for the numerical
# and the other for the catgeorical features.
# The pipeline consists of the following steps:

# For the categorical features:
mode_imputer = SimpleImputer(strategy = 'most_frequent')
encoder = OneHotEncoder(sparse_output = False)

# For the numerical features:
mean_imputer = SimpleImputer(strategy = 'mean')
scaler = StandardScaler()

# The two pipes are defined and the steps are added to them.
inner_pipe_cat = Pipeline([ ('mode', mode_imputer),('to_df', ArrayToDataFrame(cat_features)), ('ohe', encoder) ])
inner_pipe_num = Pipeline([ ('mean', mean_imputer),('scale', scaler) ])

# We unite the two pipes in a ColumnTransformer object. The parameter remainder
# is set to 'passthrough' in order to keep all the columns
# at this stage, also the ones that are not transformed/preprocessed.

transformer = ColumnTransformer([('inner_pipe_cat', inner_pipe_cat, cat_features),
                                 ('inner_pipe_num', inner_pipe_num, num_features)], 
                                remainder = 'passthrough')

# We want to test two models: RandomForest, and GradientBoosting (XGBoost).
# For each of theses models we define a hyperparameter grid that we want to 
# search for the best parameters.
# As part of the grid search a 3-fold cross validation is performed and the 
# best model is selected based on the achieved Roc_auc score on the training data,
# respectively on the validation sets that are part of the training data.
# The best model is then used to make predictions on the test data to see how 
# well the generalization on previously unseen data is working.

# First, we define a dictionary that contains the models we want to test. The 
# key is the name of the model and the value is the model itself.
# In that way, we can easily iterate over the models in the next step, when we 
# are going to look for the best performing one in 
# terms of Roc_auc score.

# We import the models we want to test.
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# We define/initalize the models we want to test in a dictionary.
models = {
    'XGBClassifier': XGBClassifier(),
    'RandomForestClassifier': RandomForestClassifier(random_state = 42) # Set a seed for reproducibility
}

# Second, we define the hyperparameter grids for the models.

# We want to test different hyperparameters for each model in order to find 
# the best performing one. In order to keep the runtime of the grid search 
# reasonable, we only test a couple of values for each hyperparameter.
# We put all grids in a dictionary, where the key is the name of the model and 
# the value is the hyperparameter grid. In that way, we can easily iterate over
#  the models and the respective hyperparameter grid in the next step.

param_grids = {
    'XGBClassifier': {
        'xgbclassifier__n_estimators': [75, 150, 300, 450],             # For the XGBR model itself we are trying different 
                                                                        # number of gradient boosted trees to be trained:
                                                                        # 75, 150, 300, and 450.
        
        'xgbclassifier__max_depth': [4, 6, 8, 12 ,14],                  # We are also testing out different maximal depths of
                                                                        # each tree: 4, 6, 8, 12, or 14 splits.
        
        'xgbclassifier__learning_rate': [0.01, 0.05, 0.1, 0.2],         # We also try different step size at each iteration 
                                                                        # while moving toward a minimum of the loss function.
                                                                        # Smaller learning rate -> more iterations to find 
                                                                        # minimum of the loss function, but also more precise 
                                                                        # estimates for this minimum.
                                                                        # A learning rate that is too large can cause the
                                                                        # algorithm to diverge and never find a good solution.
                                                                        
        'xgbclassifier__random_state': [42]                             # Set a seed for reproducibility
    },
    'RandomForestClassifier': {                                             
        'randomforestclassifier__n_estimators': [75, 100, 200, 300, 450],  # We try out different number of trees in the forest,
        'randomforestclassifier__max_depth': [6, 8, 12 ,14],               # the maximum depth of each tree,
        'randomforestclassifier__min_samples_split': [2, 20, 40],          # minimum number of instances required to make a split,
        'randomforestclassifier__min_samples_leaf': [1, 10, 20],           # minimum number of instances required to be in a node,
        'randomforestclassifier__max_features': [1.0, 'sqrt']              # and the number of features to be considered.

                                                                           # When making splitting decision: max_features = 1.0 -->
                                                                           # all features are considered
                                                                           # If max_features = "sqrt" --> squared root of the number of predictors
                                                                           # is considered.
    }

}


# Here we loop through the different models using the corresponding parameter 
# grids of the models to perform the grid search.

from sklearn.model_selection import GridSearchCV

best_params_and_best_score_per_model = {} 
# We store the best parameters, the best score, and the best estimator
#  for each model in this dictionary.
     
for model_name, model in models.items(): # We iterate over the models dictionary.
             
             # We define the current pipeline with the iteration-specific model.
             # The column transformer is the same for all models, so we do not 
             # need to redefine it in each loop. We use the transformer created
             # above, and add the current model to the pipeline.
             current_pipe = Pipeline([('preprocess', transformer), 
                                      (model_name.lower(), model)])

             # Then we perform the grid search with the current model, based on
             # the grids previously defined.As part of the gird search a 3-fold
             # cross validation is performed and the best model is selected 
             # based on the ROC_AUC score on the training data. The training 
             # data is split into 3 parts, and the model is trained on 2 parts 
             # and validated on the remaining part. This is done 3 times, so 
             # that each part is used as validation set once.
             grid_search = GridSearchCV(current_pipe,
                                        param_grids[model_name], 
                                        cv=3, 
                                        scoring='roc_auc', # We use the ROC_AUC score as the metric to evaluate the models.
                                                           # We chose the best model based on this score. 
                                        n_jobs = -2)       # We use all available cores for paralell processing expect one 
                                                           # to speed up the grid search.

             grid_search.fit(X_train, y_train)
            
             # For each loop iteration we print out the validation scores to 
             # compare the models in the end, and the best hyperparameters.
        
             print(f"{model_name} best score: {grid_search.best_score_}")
             print(f"{model_name} best params: {grid_search.best_params_}")

             # We store the best parameters, the best score, and the best model
             # object itself for each model in the dictionary. 
             best_params_and_best_score_per_model[model_name] = (grid_search.best_params_, 
                                                                 grid_search.best_estimator_, 
                                                                 grid_search.best_score_)
            
# Then we compare the models, and store the best one as a joblib object to the 
# S3 bucket.
best_model_name = None
best_estimator = None

# We start the comparison with the lowest possible score
highest_score = -float('inf')  

# We loop through the dictionary and select the model with the highest score:
for model_name, details in best_params_and_best_score_per_model.items():
    _, estimator, score = details  # Unpack the tuple
    if score > highest_score:
        highest_score = score
        best_model_name = model_name
        best_estimator = estimator

# We assign the best estimator as our final model.
model = best_estimator

# And save the trained model to the S3 bucket as a joblib object.

# First, we define the destination.
s3_client = boto3.client('s3')
bucket_name = 'final-project-team-3a'
model_key = 'model/trained_model.joblib'

# Second, we serialize the model into a bytes buffer.
buffer = BytesIO()
dump(model, buffer)
buffer.seek(0)

# And finally, we upload from the buffer to the specified S3 destination.
s3_client.upload_fileobj(buffer, bucket_name, model_key)