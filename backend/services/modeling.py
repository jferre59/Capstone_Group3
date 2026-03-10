import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import warnings
import os
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore") #Supress warnings to avoid cluttering the terminal

current_file_dir = os.path.dirname(os.path.realpath(__file__)) #Get active file directory
parent_dir = os.path.dirname(current_file_dir) #Get parent of active file directory
os.chdir(os.path.dirname(parent_dir)) #Set active directory to parent of active file directory, sets directory to backend folder
 
dataset_csv = pd.read_csv("backend/data/processed/respiratory_processed.csv") #Read the csv file containing the data into a pandas dataframe

imputer = SimpleImputer(missing_values=np.nan, strategy='median') #Simple imputer instance to imput median into missing column values

features = dataset_csv.drop(columns=['age', 'treatment', 'disease', 'sex', 'nature', 'age_group'], axis=1) #Get all feature data without labels and columns which have no bearing on results
df_disease_columns = features.filter(like='disease', axis=1)
df_treat_columns = features.filter(like='treatment', axis=1)
df_symp_columns = features.filter(like='symptom', axis=1)
features = features.drop(columns=df_disease_columns.columns, axis=1) #Remove disease columns, contain encoded label data
features = features.drop(columns=df_treat_columns.columns, axis=1) #Remove treatment column, treatment will not be known until after diagnosis
features = features.drop(df_symp_columns.columns, axis=1) #Remove treatment column, treatment will not be known until after diagnosis
features['symptoms'] = dataset_csv['symptoms'].astype('category').cat.codes
imputed_features = imputer.fit_transform(features) #Impute column median into missing value spaces

labels = dataset_csv['disease'] #Get labels data without feature data
mode_value = labels.mode()[0] #Get the mode of the labels
labels.fillna(mode_value, inplace=True) #Fill missing values with mode

X_train, X_test, y_train, y_test = train_test_split(imputed_features, labels, test_size=0.3, stratify=labels, random_state=42) #Perform train test split with 85% for training and 15% for testing

print(type(X_test))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42) #Best model found after grid search results 

rf_model.fit(X_train, y_train) #train model on data 

y_pred = rf_model.predict(X_test) #Run predictions on testing data
print(y_pred)

accuracy = accuracy_score(y_test, y_pred) #Calculate accuracy score and print
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}") #Calculate and print f1 score
filename = 'backend/model/trained_model.joblib'

joblib.dump(rf_model, filename)