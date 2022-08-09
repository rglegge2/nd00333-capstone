from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run


def clean_data(df):    
    # Drop unused columns
    x_df = df.dropna()
    x_df = x_df.drop(['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)
    
    # Clean and one hot encode data
    x_df['Gender'] = x_df['Gender'].apply(lambda s: 1 if s == 'M' else 0)
    education = pd.get_dummies(x_df['Education_Level'], prefix='Education_Level')
    x_df.drop('Education_Level', inplace=True, axis=1)
    x_df.join(education)
    marital = pd.get_dummies(x_df['Marital_Status'], prefix='Marital_Status')
    x_df.drop('Marital_Status', inplace=True, axis=1)
    x_df.join(marital)
    income = pd.get_dummies(x_df['Income_Category'], prefix='Income_Category')
    x_df.drop('Income_Category', inplace=True, axis=1)
    x_df.join(income)
    card = pd.get_dummies(x_df['Card_Category'], prefix='Card_Category')
    x_df.drop('Card_Category', inplace=True, axis=1)
    x_df.join(card)
    
    y_df = x_df.pop('Attrition_Flag').apply(lambda s: 1 if s == 'Existing Customer' else 0)
    return x_df, y_df
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength. Smaller values cause stronger regularization')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations to converge')
    args = parser.parse_args()
    run = Run.get_context()
    run.log('Regularization Strength: ', np.float(args.C))
    run.log('Max iterations: ', np.int(args.max_iter))
    
    df = pd.read_csv('./credit-card-bank-churn/credit_card_churn.csv')
    
    # Clean dataset
    x, y = clean_data(df)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    run.log('AUC', np.float(auc))

if __name__ =='__main__':
    main()