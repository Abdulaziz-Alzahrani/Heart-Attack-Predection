# Heart-Attack-Predection

## 1. Project Overview:

A heart attack is the block of blood flow, oftenly this blockage is caused by substances such as fat and cholesterol, Which form what is called
a plaque in the arteries that feed the heart. Heart attacks can be diagnosed using various methods of test such as Heart-specific diagnostic tests,
Blood tests, History and symptoms and Imaging tests.
This Project aims to build a model that can help predict if the patient has a high or low chance of having a heart attack based on 
some medical tests taken from the patient such as cholesterol levels and resting blood pressure also iformation about the patients
such as age and sex are used.

## 2. Problem Statement
Heart attacks can suddenly occure and as we know the results can be fetal or can cause a permanent damage, so the idea of a machine
learning model that can give the person an indication about his heart health status and to do that we are going to use various machine
learning algorthims some of them are shallow such as Logistic Regression and SVM and ensemble such as Random Forest and try to find 
the best parameters for each then train and test theses models followed by the process of evaluation of the models using confusion
matrix, accuracy, precision, recall and f1-score.   
## 3. Metrics
In the evaluation process confusion matrix would be used to get the values of TP, FP, TN and FN which are going to be used in all
other metrics starting from the accuracy ending with f1-score.
The accuracy can be an overview on the performance of the model and it is calculated as:

accuracy = (TP + TN) / (TP + TN + FP + FN)

Also precision and recall would be used as precision is the measure of how many correct cases with low chance of heart attack did
the model predict correctly and recall is the measure of how many correct cases with high chance of heart attack did
the model predict correctly. In our case recall is the most important metric in this project as misclassification of patient with
a high chance of a heart attack can put him at risk. The F-score is a measure of a test's accuracy. It is calculated from the precision 
and recall of the test.

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1-score = (2 * precision * recall) / precision + recall

## 4. Analysis
###     4.1 Data Exploration 
The dataset was obtained from [Kaggle](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) and it is a comma seperated file **(CSV)** containing 303 records and made up of 14 features.
The dataset did not contain any missing value and contained only one duplicated value thus leaving us with 302 records.
###     4.2 Data Visualization 
Some visualizations were made and can be seen in the following:
![alt text](App/static/imgs/Figure_1.png?raw=true "relation between age and chance of a heart attack")
As it can be seen the ages are between 29 and 77 also the output is represented in 0 and 1 where 0 is a low chance of heart
attack and 1 is a high chance of heart attack, on the other hand the chances gets high in 40's to mid 50's.
![alt text](App/static/imgs/Figure_2.png?raw=true "relation between sex and chance of a heart attack")
In previous figure we can see the relation between sex and the chances of a heart attack snd sex is represented also by 0 and 1
, 0 for male and 1 for female and we can conclude that males have a higher chance getting a heart attack.
## 5. Methodology
###     5.1 Data Preprocessing 
Some data preorocessing were conducted starting by checking for missing values and if there is indeed a missing value it will be
filled with the average value of that column also duplicate values are droped and then saved to a SQLite db.
###     5.2 Implementation 
The data is read from the db file then gets splitted to into training/ testing sets where the testing set gets 10% of data. After
that a Logistic Regression, SVC, KNN and a Random forest models are built after the that the models are trained then evaluated and 
metrics for each model is saved in a folder including the model itself in a pickle format
###     5.3 Refinement 
GreadSearchCv was used to find the best parameters for each model and a cross validation with a 4 k-folds and the following is 
the hyperparameters used for each model:
```bash
    params = [
        {'max_iter': [100, 200, 300], 'penalty':['l1', 'l2', 'none'], 'C':[0.1, 0.5, 1.0]}, #LogisticRegression
        {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[0.1, 0.5, 1, 5, 10]},  #SVC
        {'n_neighbors':[*range(1,67)], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},  #KNN
        {'n_estimators':[100, 200, 300, 400, 500], 'criterion': ['gini', 'entropy']}    #RandomForestClassifier
    ]
```
## 6. Results
###     6.1 Model Evaluation and validation 
####    a) Logistic Regression:
            Confusion matrix:
![alt text](Model/model-0/model.png?raw=true)
            
            Metrics:

                accuracy: 0.74
                precision: 0.65
                recall: 1.0
                F1: 0.79
```bash
{'C': 0.1, 'max_iter': 100, 'penalty': 'l2'}
```
####    b) SVC:
            Confusion matrix:
![alt text](Model/model-1/model.png?raw=true)
            
            Metrics:

                accuracy: 0.74
                precision: 0.65
                recall: 1.0
                F1: 0.79
```bash
{'C': 0.1, 'kernel': 'linear'}
```
####    c) KNN:
            Confusion matrix:
![alt text](Model/model-2/model.png?raw=true)
           
           Metrics:

                accuracy: 0.67
                precision: 0.61
                recall: 0.86
                F1: 0.72
```bash
{'algorithm': 'auto', 'n_neighbors': 11}
```
####    d) Random Forest:
            Confusion matrix:
![alt text](Model/model-3/model.png?raw=true)
           
           Metrics:

                accuracy: 0.80
                precision: 0.73
                recall: 0.93
                F1: 0.82
```bash
{'criterion': 'entropy', 'n_estimators': 100}
```
###     6.2 Justification 
Since the data was low grid search was used to find the best parameters and as it can be seen how the results varied from model to model
and if had to choose between models I would suggest eather Logistic Regression or SVC as there results are similar and most importantly
a 1.0 recall. I would guess that the reason that Random forest was not hte top as it was complicated with low data causing an underfitting.
## 7. Conclusion
###     7.1 Reflection 
To conclude this project as it started from obtaining the data set, exploring it, preprocessing and modeling using multiple algorthims and grid search then evaluting them using diffrent metrics. The most interestin g part in this project was seeing the results obtained from Random Forest
as I assumed it would perform the best.
###     7.2 Improvement 
I was really expecting lower performane consedirng the low data in the dataset and in terms of performance I would try more parameters in the
grid search also by trying other algorithms such as nueral networks and I hope the dataset would get bigger.

## Project Content
### Heart-Attack-Predection
The project folder that contains the following
#### app folder:
This folder contains "app.py" where the flask app code is written and "templates" folder which contains the html pages
required by the app.
#### data folder:
This folder containes the dataset and the "data_cleaning.py" which does the data preprocessing and saves it in a SQLite db also there
is "analysis.py" where data some data analysis and visualizations are done.
#### model folder:
This folder containes "train_classifier.py" which builds models stores it in a pickle format including metrics of each model.
#### Readme.md file:
A discreption of the project.
#### Licence:
MIT Licence.

## How to Use the Project:
Before showing how to use the scripts lets go through the required libraies:
### 1- panadas
```bash
pip install pandas
```
### 2- numpy
```bash
pip install numpy
```
### 3- scikit-learn
```bash
pip install scikit-learn
```
### 4- SQLAlchemy
```bash
pip install sqlalchemy
```
### 5- flask
```bash
pip install flask
```

### 6- matplotlib
```bash
pip install matplotlib 
```

### 7- seaborn
```bash
pip install seaborn
```
To use this project you will have to go the Data directory and run the follwing command:
```bash
python data_cleaning.py
```
this will preprocess the data and save it to a db file inside the Models directory then
go ahead and move to the Model directory and run the following command: 
```bash
python train_classifier.py
```
Now you can go to app directory and run the flask web app using run the following command.
```bash
python app.py
```
To open the project on your browser copy the link from your cmd next to:
 _* Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)_
 
 and paste in your browser, then you can use the app to make predictions.
 
 ## Acknowledgements 

I would like to express my gratitude to Misk Academy and Udacity for this Amazing program
that expanded my knowledge and helped me in making this project.
