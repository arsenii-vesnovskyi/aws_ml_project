# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:34:55 2024

@author: arsen
"""
import pandas as pd
import requests
import json

# First, we define the URL where our Flask app is running
url = 'http://54.172.34.205:5000/predict'


# Then, we load our test data. In the normal scenario, here the user would 
# provide the new data they want to make the predictions for. We are using test
# data here in order to see how good out model generalizes to the unseen data.
df_new_data = pd.read_csv(r"C:\Users\arsen\OneDrive\Документы\ESADE\08_Second Term\Cloud Platforms AWS\Final Project\model_janick\test_data.csv")

# For the data that is going to be sent to the model for predictions we drop
# the columns with real values. This step is not necessary for completely new
# data. The conversion to dict is still necessary.
data = df_new_data.drop(columns = ["Attrition_Flag"]).to_dict(orient='list')

# After that, we convert the dictionary to a JSON string before sending it to 
# the API.
data_json = json.dumps(data)

# We make the POST request to the API and save the response.
response = requests.post(url, data=data_json, headers={'Content-Type': 'application/json'})

# As the next step, we extract the actual data from the response stored under 
# key "result" as we defined it in the API.
result_str = response.json()['result']

# Then, we convert the JSON formatted string back into a Python object 
# (list of dicts in this case).
result_data = json.loads(result_str)

# And finally, we convert the list of dicts to a DataFrame.
predictions_df = pd.DataFrame(result_data)

print(predictions_df)


# We also use this script to evaluate the predictions of the model on the test
# set:

predictions = predictions_df['prediction'] 

real_values = df_new_data['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
  
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, precision_score

roc_auc = roc_auc_score(real_values, predictions)

accuracy = accuracy_score(real_values, predictions)

recall = recall_score(real_values, predictions)

precision = precision_score(real_values, predictions)

conf_matrix = confusion_matrix(real_values, predictions)

print("ROC AUC Score:", roc_auc)
print("Accuracy Score:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("Confusion Matrix:\n", conf_matrix)