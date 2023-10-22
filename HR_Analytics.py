#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:42:11 2023

@author: yonastena
"""

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data=pd.read_csv('HR_data.csv')
data.head()

#check for missing values in the dataset
data.isna().sum()

#check for duplicates
len(data_[data_.duplicated()==True])

#drop duplicates
data_1=data.drop_duplicates(keep='first')

#renaming columns
data_1=data_1.rename(columns=
    {
    'time_spend_company':'tenure',
    'average_montly_hours': 'average_monthly_hours',
    'Work_accident':'work_accident',
    'Department':'department'
    
    })

#check for outliers in the column 'tenure'
percentile_25=data_1['tenure'].quantile(0.25)
percentile_75=data_1['tenure'].quantile(0.75)
IQR=percentile_75-percentile_25

lower_threshold=percentile_25-1.5*IQR
upper_threshold=percentile_75+1.5*IQR

outliers_tenure=data_1[(data_1['tenure'] < lower_threshold)| (data_1['tenure'] > upper_threshold)]
len(outliers_tenure)

#Visualization
fig, ax = plt.subplots(1, 2, figsize = (22,8))

sns.boxplot(data=data_1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects')

sns.histplot(data=data_1,x='number_project',hue='left',multiple='dodge',shrink=2,ax=ax[1])
ax[1].set_title('Number of project vs employees left')
plt.show();


plt.figure(figsize=(16, 9))
sns.scatterplot(data=data_1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=160, color='#ff6361', label='160 hrs./mo.', ls='--')
plt.legend(labels=['160 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');


sns.catplot(data=data_1,y='satisfaction_level',x='tenure',hue='left',kind='violin',split=True)


data_1.groupby(['left'])[['satisfaction_level']].agg([np.median,np.mean])
data_1.groupby(['left'])[['last_evaluation']].agg([np.median,np.mean])
data_1.groupby(['salary'])[['satisfaction_level']].agg([np.median,np.mean])

#Logistic model
print(data_1['left'].nunique())


log_data=data_1.drop(outliers_tenure.index)  # one important assumption of logistic model is no extreme values

#Data pre-processing 
#most machine learning models can't handle categorical features so categorical columns need to be converted to numeric
log_data['salary'].value_counts()
log_data['department'].value_counts()
log_data_encoded=log_data.copy()

log_data_encoded['salary'] = (
    log_data_encoded['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

log_data_encoded['salary'].value_counts()
log_data_final=pd.get_dummies(log_data_encoded,drop_first=True)

#Heatmap of logistic data to check multicollinearity among predictors

plt.figure(figsize=(8, 6))
sns.heatmap(log_data_final[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure','work_accident','promotion_last_5years']].corr()
                                ,annot=True,cmap='crest')
plt.title('Heatmap of the logistic dataset')
plt.show()

#split the data in to training and testing set
X=log_data_final.copy()
X=X.drop(['left'],axis=1)
y=log_data_final['left']
log_data_final['left'].value_counts(normalize=True)# checks whether it is an imbalanced data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
#build logistic classifier model
log_model=LogisticRegression(max_iter=500).fit(X_train,y_train)
log_model.coef_
log_model.intercept_
#Use the logistic regression model to get predictions on the test dataset
y_pred=log_model.predict(X_test)
log_model.predict_proba(X_test)
#confusion matrix to visualize the results of the logistic model
log_cm=confusion_matrix(y_test,y_pred,labels=log_model.classes_)
log_model_disp=ConfusionMatrixDisplay(log_cm,display_labels=log_model.classes_)
log_model_disp.plot(values_format='')
plt.show()

#classification report
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))

#Evaluate the results of the logistic model
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))
print("AUC Score:", "%.6f" % metrics.roc_auc_score(y_test, y_pred))


data_tree=data_1.copy()
data_tree['salary']=data_tree['salary'].replace(['low','medium','high'],[0,1,2])
data_tree['salary'].value_counts()

data_tree=pd.get_dummies(data_tree,drop_first=True)

X=data_tree.drop('left',axis=1)
y=data_tree['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
# Instantiatethe decisiontree model
treeclf = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, None],
             'min_samples_leaf': [4, 6],
             'min_samples_split': [4, 6]
             }
# Assign a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Instantiate GridSearch
dt = GridSearchCV(treeclf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

%%time

dt.fit(X_train, y_train)
dt.best_params_
dt.best_score_

# 1. Instantiate the random forest classifier
rfclf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  

# 3. Define a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1','roc_auc']

# 4. Instantiate the GridSearchCV object
rf = GridSearchCV(rfclf, cv_params, scoring=scoring, cv=4, refit='roc_auc')


#Fit the random forest model to the training data.
%%time
rf.fit(X_train, y_train)

# Define a path to the folder where you want to save the model
path = '/Users/yonastena/Downloads/Google DA'

def write_pickle(path, model_object, save_as:str):
    '''
    In: 
        path:         path of folder where you want to save the pickle
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the folder indicated
    '''    

    with open(path + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)

def read_pickle(path, saved_model_name:str):
    '''
    In: 
        path:             path to folder where you want to read from
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model

# Write pickle
write_pickle(path, rf, 'hr_rf')

# Read pickle
rf = read_pickle(path, 'hr_rf')

#check the best score roc_auc in this case and compare it with the dt
rf.best_score_

# Check best params
rf.best_params_

#compare the results of the decision tree and random forest using a table format
def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })
  
    return table

dt_result = make_results('decision tree', dt, 'auc')
dt_result

rf_result = make_results('random forest', rf, 'auc')
rf_result

results = pd.concat([dt_result, rf_result], axis=0)
results

#Next step is to evaluate the best random forest model on the test set.
#In order to do that we need to define a function that gets all the scores from a model's predictions.
def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                         })
  
    return table

# Get predictions on test data
rf_test_score = get_scores('random forest test', rf, X_test, y_test)
rf_test_score

#final result
final_result = pd.concat([results, rf_test_score], axis=0)
final_result

# Generate array of values for confusion matrix
pred = rf.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, pred, labels=rf.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf.classes_)
disp.plot(values_format='');

#Feature Importnace of the best selected model
# Get feature importances
feat_impt = rf.best_estimator_.feature_importances_

# Get indices of top 10 features
ind = np.argpartition(rf.best_estimator_.feature_importances_, -10)[-10:]

# Get column labels of top 10 features 
feat = X.columns[ind]

# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]

y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)

y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")




###Bonus for plotting decision trees

from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

treeclf = DecisionTreeClassifier(random_state=42, max_depth=4)
treeclf.fit(X_train,y_train)

dot_data = StringIO()
export_graphviz(treeclf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = X.columns,class_names=['stayed','left'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('hr.png')
Image(graph.create_png())

dftree=pd.DataFrame({'Feature_names':X.columns,'Importances':treeclf.fit(X_test,y_test).feature_importances_})
dftree

dftree=dftree.sort_values(by='Importances',ascending=False)
sns.barplot(data=dftree,x='Importances',y='Feature_names',orient='h')
#since we have many 0 values in the 'Importances' column, let's remove these values
dfdt=dftree[dftree['Importances']!=0]
dfdt
sns.barplot(data=dfdt,x='Importances',y='Feature_names',orient='h')
