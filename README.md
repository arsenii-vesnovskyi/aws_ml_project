**Project Description**

The current repository contains the files for the group project implementation of a bank churn prediction model trained on an AWS EC2 cloud instance and served for making predictions via a flask API endpoint that ran on the same instance. The training and test data, as well as the model itself (as a joblib file) were saved on in the S3  storage. The storage was also connected to the AWS QuickStart Dashboarding Service to visualize the data (incl. predictions).
Due to the recurring costs of AWS the API endpoint is currently not available. However, the purpose of the repository is to showcase the project itself. Glad to answer any questions in that regard!

**File Tree**

* model_train.py - Script run on EC2 to train the model and save it to an S3 bucket
* trained_model.joblib - Trained model saved on the same S3
* predictions_api.py - Script for the API that is run on EC2 that makes predictions with the model and uploads new data for analysis to the S3 bucket
* request_to_api.py - Script that is to be executed locally to access the API and obtain predictions for test data, and then analyze the performance of the model
* quickstart-manifest-analysis.json - JSON file that is needed to point AWS QuickStart Dashboard to the analysis file in the S3 bucket
* training_data.csv - the data used to train the model
* test_data.csv - the data used to train the model
* Architecture Diagram.png - the visualization of the project's architecture

**Link to Demo**

https://drive.google.com/file/d/1nCDGUvQSov5a307komKywS2ZucgoUTv1/view?usp=drive_link
