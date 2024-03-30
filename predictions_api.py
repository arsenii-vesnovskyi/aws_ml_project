# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:47:59 2024

@author: arsen
"""
# First, we import the needed libraries
from flask import Flask, request, jsonify
import pandas as pd
import boto3
from joblib import load
from io import BytesIO, StringIO
import traceback
from sklearn import set_config

# We also set configuration to transform any non-pandas output in the pipe steps
# to the pandas DataFrame.
set_config(transform_output="pandas")

# However, as testing out different set-ups showed, the previous setting does not
# always guarantee the needed effect. That is why, in the model_train.py script
# we deined the to_df step in the pipeline. In order for this custom step to 
# function in the API script, we need to redefine it here.

from sklearn.base import BaseEstimator, TransformerMixin

class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.column_names)
    
# Then, we can proceed with the definition of the flask app.

app = Flask(__name__)

# We define the function to load our model from S3 storage.
def load_model_from_s3(bucket_name, model_key):
    s3_client = boto3.client('s3')
    with BytesIO() as f:
        s3_client.download_fileobj(bucket_name, model_key, f)
        f.seek(0)
        model = load(f)
    return model

# In the next step, we specify the exact location of the model joblib file.
bucket_name = 'final-project-team-3a'
model_key = 'model/trained_model.joblib'

# And after that, we are ready to load our trained model into the script.
model = load_model_from_s3(bucket_name, model_key)

# We define the endpoint and the post method for it. Post method is needed in
# order to provide data to the API on which predictions will be made.

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # We extract JSON data from request
        data = request.get_json()
        
        # If needed we log received data in order to inspect it.
        # app.logger.debug(f"Received data: {data}")  
        
        # Then, we convert data into DataFrame
        df = pd.DataFrame(data)

        # And inspect its structure if needed as well.
        # print("DataFrame dtypes:\n", df.dtypes)
        # print("DataFrame columns:\n", df.columns)
        
        # Then, we make the predictions with our model.
        predictions = model.predict(df)
        
        # And append them as a new column to the DataFrame.
        df['prediction'] = predictions
    
        # Before returning the prediction to the user, we want to append the new
        # data and predictions to the analysis file on the S3 in order for it
        # to be accessible via dashboard.
        
        analysis_data = pd.read_csv('s3://final-project-team-3a/analysis/analysis_data.csv')
        
        # First, we rename column "prediction" to "Attrition_Flag" to follow the
        # structure of analysis data file.
        new_data = df.rename(columns={'prediction': 'Attrition_Flag'})
        new_data['Attrition_Flag'] = new_data['Attrition_Flag'].map({ 0: 'Existing Customer', 
                                                   1 : 'Attrited Customer'})
        # Second, we add a new column "predicted" with all values set to 1, which
        # indicates that the values are not real, but predicted. In the training
        # data this column is filled with 0.
        new_data['predicted'] = 1

        # Third, we append new_data to analysis_data row-wise.
        analysis_data = pd.concat([analysis_data, new_data], ignore_index=True)
        
        # And upload it back to the S3.
        
        # First, we convert DataFrame to a csv format.
        csv_buffer = StringIO()
        analysis_data.to_csv(csv_buffer, index=False)
        
        # And create a boto3 client.
        s3_client = boto3.client('s3')
        
        # Then, we specify the file path.
        file_path = 'analysis/analysis_data.csv'
        
        # And upload the updated csv to S3
        s3_client.put_object(Bucket=bucket_name, Key=file_path, Body=csv_buffer.getvalue())

        # And after that, we can convert the modified DataFrame with predictions
        # that we want to return to the user to JSON format.
        result = df.to_json(orient="records")
        
        # And return the JSON response with predictions to the client.
        return jsonify({'result': result}), 200
    except Exception as e:
        # In case any of the previous steps failed, we print out the trace to 
        # the error.  
        traceback.print_exc()
        
        # And save the log of the exception with that trace.
        app.logger.error('Failed to make a prediction', exc_info=True)
        
        # Finally, we return the exception as a json response to the client.
        return jsonify({'error': str(e)}), 500
    

# Then, we run the flask API app on the needed port.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
