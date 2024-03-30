Current GitHub repository contains the following files:

* model_train.py - Script run on EC2 to train the model adn save it to S3
* trained_model.joblib - Trained model save on S3
* predictions_api.py - Script for the API tun on EC2 that makes predictions with the model and uploads new data for analysis to S3
* request_to_api.py - Script execured locally to access the API and obtain predictions for test data, and then analyze the performance of the model
* quickstart-manifest-analysis.json - JSON file that is needed to point AWS QuickStart Dashboard to the analysis file in S3 bucket
