#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def read_file(filename):
    '''given the name of a csv file (string),
        read in the contents and return as a 2d list
        of strings '''
    data = pd.read_csv(filename)
    print(data.columns)
    return data

def cross_fold_validation(data):
    ''' given a data frame, performs cross fold validation to 
    find the optimal k value based on accuracy and returns a list 
    containing the mean scores, the best k values for 
    the different scoring metrics and some information about the highest 
    and lowest mean scores 
    '''
    k_values = range(4, 13)
    mean_scores = {'accuracy':[], 'precision':[], 'recall':[]}
    X = data.drop(["Sex", "Heart Disease"], axis=1)
    y = data["Heart Disease"]
    

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
      
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cross = cross_validate(model, X, y, cv=kf, 
                scoring=['accuracy', 'precision_macro', 'recall_macro'])

        mean_accuracy = np.mean(cross['test_accuracy'])
        mean_precision = np.mean(cross['test_precision_macro'])
        mean_recall = np.mean(cross['test_recall_macro'])
    
        mean_scores['accuracy'].append(mean_accuracy)
        mean_scores['precision'].append(mean_precision)
        mean_scores['recall'].append(mean_recall)

    best_accuracy_k = np.argmax(mean_scores['accuracy']) + 4
    best_precision_k = np.argmax(mean_scores['precision']) 
    best_recall_k = np.argmax(mean_scores['recall']) + 4
    
    best_k_values = [best_accuracy_k, best_precision_k, best_recall_k, 
                     mean_scores]
    return best_k_values

def classifer(data, best_k_values):
    '''given a data frame and list containing best_k_values, label maps the y 
    values and splits the X and y into training data, builds a knn
    classifier for the different scoring metrics and returns the 
    scores of each model along with the precision classifer for the user
    model
    '''
    X = data.drop(["Sex", "Heart Disease"], axis=1)
    label_mapping = {'Absence': 0, 'Presence': 1}
    y = data["Heart Disease"].map(label_mapping)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    accuracy_k, precision_k, recall_k, mean_scores = best_k_values
    
    # Classifier for accuracy
    classifier_accuracy = KNeighborsClassifier(n_neighbors=accuracy_k)
    classifier_accuracy.fit(X_train, y_train)
    y_pred_accuracy = classifier_accuracy.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_accuracy)
    
    # Classifier for precision
    classifier_precision = KNeighborsClassifier(n_neighbors=precision_k)
    classifier_precision.fit(X_train, y_train)
    y_pred_precision = classifier_precision.predict(X_test)
    precision = precision_score(y_test, y_pred_precision)
    
    # Classifier for recall
    classifier_recall = KNeighborsClassifier(n_neighbors=recall_k)
    classifier_recall.fit(X_train, y_train)
    y_pred_recall = classifier_recall.predict(X_test)
    recall = recall_score(y_test, y_pred_recall)
    
    knn_scores = [accuracy, precision, recall, classifier_precision]
    return knn_scores
    

def logistic_regression(data):
    '''given a data frame, label maps the y values and splits the X and y into 
    training data, builds a logistic regression model and returns the accuracy,
    precision, and recall scores along with the model for the user model
    '''
    X = data.drop(["Sex", "Heart Disease"], axis=1)
    label_mapping = {'Absence': 0, 'Presence': 1}
    y = data["Heart Disease"].map(label_mapping)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    lr_scores = [accuracy, precision, recall, model]
    return lr_scores

def compare_models(knn_scores, lr_scores):
    '''given a list of the knn scores and the logistic regression scores, 
    plots the scores of each model based on accuracy, precision, and recall
    in a seaborn barplot
    '''
    models = ["K-Nearest Neighbors", "Logistic Regression"]
    metrics = ["Accuracy", "Precision", "Recall"]
    
    # Extracting values from knn_scores and lr_scores
    knn_accuracy, knn_precision, knn_recall, classifer_precision = knn_scores
    lr_accuracy, lr_precision, lr_recall, model= lr_scores
    
       
    data = {
    "Model": models * 3,
    "Metric": metrics * 2,
    "Score": [knn_accuracy, knn_precision, knn_recall, lr_accuracy,
              lr_precision, lr_recall]
    }
    df = pd.DataFrame(data)
    
    # Plotting using seaborn barplot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Metric", y="Score", hue="Model", data=df, 
                     palette="Set2")
    plt.title('Comparison of Metrics between kNN and Logistic Regression')
    plt.ylabel('Score')
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 10), 
                    textcoords = 'offset points')
    plt.legend(loc='upper right', fontsize='small')

    plt.show()
    

def predict(knn_scores, lr_scores):
    '''given a list of knn_scores containing the precision classifer 
    (best scoring model) and a list of lr_scores containing the model,
    creates a user accessible feature in which users can input their own
    data and choose the scoring metric they care most about to predict the 
    presence of heart disease.
    '''
    knn_accuracy, knn_precision, knn_recall, classifer_precision = knn_scores
    lr_accuracy, lr_precision, lr_recall, model = lr_scores
    
    print("Please provide the following information to predict heart disease:")

    age = float(input("Age: "))
    chest_pain_type = float(input("Chest Pain Type (1: typical angina,"\
                                  " 2: atypical angina, 3: non-anginal pain,"\
                                      " 4: asymptomatic): "))
    bp = float(input("Blood Pressure: "))
    cholesterol = float(input("Cholesterol: "))
    
    fbs = float(input("Fasting Blood Sugar \
                      (> 120 mg/dl, 1 for True, 0 for False): "))
    ekg_results = float(input("EKG Results (0: normal, \
                1: having ST-T wave abnormality, 2: showing probable or \
                    definite left ventricular hypertrophy): "))
    max_hr = float(input("Max Heart Rate: "))
    exercise_angina = float(input("Exercise-Induced Angina "\
                                  "(1 for Yes, 0 for No): "))
    st_depression = float(input("ST Depression: "))
    slope_st = float(input("Slope of ST segment (0: upsloping, 1: flat,"\
                           " 2: downsloping): "))
    num_vessels_fluroro = float(input("Number of Major Vessels Fluoroscopy: "))
    thalium = float(input("Thalium Stress Test Result (3: normal, "\
                          "6: fixed defect, 7: reversible defect): "))
    

    print("\nWhich scoring metric is most important to you?")
    print("1. Accuracy")
    print("2. Precision")
    print("3. Recall")
    
    chosen_metric = input("Enter the number corresponding to your choice: ")
    X_data = np.array([age, chest_pain_type, bp, cholesterol, fbs,
                       ekg_results, max_hr, exercise_angina, st_depression, 
                       slope_st, num_vessels_fluroro, thalium]).reshape(1, -1)
    feature_names = ['Age', 'Chest pain type', 'BP', 'Cholesterol', 
                     'FBS over 120',
           'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
           'Slope of ST', 'Number of vessels fluro', 'Thallium']
    
    X = pd.DataFrame(X_data, columns=feature_names)
    
    if chosen_metric == '1' or chosen_metric == '3':
        y_pred = model.predict(X)
        if y_pred == 1:
            print("Model predicted presence of heart disease :(")
        else:
            print("Model predicted absense of heart disease :)")
    elif chosen_metric == '2':
        y_pred = classifer_precision.predict(X)
        if y_pred == 1:
            print("Model predicted presence of heart disease :(")
        else:
            print("Model predicted absense of heart disease :)")

    else:
       print("Invalid choice. Please enter a number from 1 to 3.")
    
def main():
    '''calls all the functions for visualizations and initiates the user 
    friendly model
    '''
    data = read_file("Heart_Disease_Prediction.csv")
    best_k = cross_fold_validation(data)
    knn = classifer(data, best_k)
    regression = logistic_regression(data)
    model_comparison = compare_models(knn, regression)
    predict(knn, regression)

if __name__ == "__main__":
    main()
    
