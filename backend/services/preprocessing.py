#Importing libraries
print("Importing libraries...")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import joblib
print("Libraries imported successfully.")

class Preprocessor:
    """
    Handles preprocessing of respiratory diagnostic data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.symptom_encoder = None
    
    def load_and_clean(self, filepath):
        """
        Initially loads data and cleans for use.
        """

        #Loading
        print(f"\nLoading data from {filepath}...")
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")

        #Handling column naming
        print("\nStandardizing column names...")
        df.columns = df.columns.str.strip().str.lower()
        print("Column names standardized.")

        #Handling missing ages
        print("\nHandling missing ages...")
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        print("Missing ages handled.")

        #Standardizing text data
        print("\nStandardizing text data...")
        df['symptoms'] = df['symptoms'].str.strip().str.lower()
        df['sex'] = df['sex'].str.strip().str.lower()
        df['disease'] = df['disease'].str.strip().str.lower()
        df['treatment'] = df['treatment'].str.strip().str.lower()
        df['nature'] = df['nature'].str.strip().str.lower()
        print("Text data standardized.")

        return df
    
    def handle_missing_values(self, df):
        """Handles missing values in the dataset."""

        print("\nHandling missing values...")

        df_processed = df.copy()

        #Age imputation using median grouped by disease
        age_imputer = df_processed.groupby('disease')['age'].transform('median')
        df_processed['age'].fillna(age_imputer, inplace=True)
        df_processed['age'].fillna(df_processed['age'].median(), inplace=True) #Using overall median if no disease group

        #Marking missing Sex as 'unknown'
        df_processed['sex'].fillna('unknown', inplace=True)

        #Removing records with missing symptoms
        df_processed = df_processed.dropna(subset=['symptoms'])

        print("Missing values handled.")

        return df_processed
    
    def encode_symptoms(self, df):
        """Uses multi-hot encoding as symptoms can be multiple per patient."""

        print("\nEncoding symptoms...")

        #Getting unique symptoms
        all_symptoms = set()
        for symptom_str in df['symptoms']:
            symptoms = [s.strip() for s in str(symptom_str).split(',')]
            all_symptoms.update(symptoms)

        #Creating binary columns per symptom
        for symptom in all_symptoms:
            df[f'symptom_{symptom}'] = df['symptoms'].apply(
                lambda x: 1 if symptom in str(x).split(',') else 0
            )
        
        print("Symptoms encoded.")

        return df, list(all_symptoms)
    
    def encode_categorical(self, df):
        """Encodes categorical variables"""
        
        print("\nEncoding categorical variables...")
        
        df_encoded = df.copy()

        #Encoding Sex (male=0, female=1, unknown=2)
        sex_mapping = {'male': 0, 'female': 1, 'unknown': 2}
        df_encoded['sex_encoded'] = df_encoded['sex'].map(sex_mapping)

        #Encoding Nature (low=0, medium=1, high=2)
        nature_mapping = {'low': 0, 'medium': 1, 'high': 2}
        df_encoded['nature_encoded'] = df_encoded['nature'].map(nature_mapping)

        #One-hot encoding disease and treatment
        disease_dummies = pd.get_dummies(df_encoded['disease'], prefix='disease')
        treatment_dummies = pd.get_dummies(df_encoded['treatment'], prefix='treatment')

        df_encoded = pd.concat([df_encoded, disease_dummies], axis=1)
        df_encoded = pd.concat([df_encoded, treatment_dummies], axis=1)

        print("Categorical variables encoded.")

        return df_encoded
    
    def feature_engineering(self, df):
        """Creating additional features based on existing data."""

        print("\nPerforming feature engineering...")

        df_features = df.copy()

        #Defining age groups
        df_features['age_group'] = pd.cut(
            df_features['age'], 
            bins=[0, 12, 18, 35, 50, 65, np.inf], 
            labels=['child', 'teen', 'young_adult', 'adult', 'middle_aged', 'senior']
        )

        #Encoding age_group
        age_group_dummies = pd.get_dummies(df_features['age_group'], prefix='age_group')
        df_features = pd.concat([df_features, age_group_dummies], axis=1)

        #Symptom count feature
        df_features['symptom_count'] = df_features['symptoms'].apply(
            lambda x: len([s.strip() for s in str(x).split(',')])
        )

        #Feature flagging high-risk
        df_features['high_risk'] = (
            ((df_features['age'] < 12) | (df_features['age'] > 65)) &
            (df_features['nature'] == 'high')
        ).astype(int)

        print("Feature engineering completed.")

        return df_features
    
    def scale_features(self, df, feature_cols=['age', 'symptom_count']):
        """Normalizinf numerical features for stability in model."""

        print("\nNormalizing numerical features...")

        df_normalized = df.copy()
        df_normalized[feature_cols] = self.scaler.fit_transform(df_normalized[feature_cols])

        print("Numerical features normalized.")

        return df_normalized
    
    def prepare_for_model(self, df):
        """Selecting features for input to model, dropping original columns not needed."""

        print("\nPrepping data for model input...")

        #Identifying feature cols (excluding original text and target)
        feature_cols = [col for col in df.columns if col.startswith(
            ('symptom_', 'age', 'sex_encoded', 'nature_encoded',
             'age_group_', 'symptom_count', 'high_risk')
        )]

        #Defing X and y
        X = df[feature_cols]
        y = df['disease']

        print("Data prepared for model input.")

        return X, y, feature_cols
    
    def pipeline(self, filepath):
        """Executes full preprocessing pipeline."""

        print("\nStarting preprocessing pipeline...")
        print("1/7 -- Loading data --")
        df = self.load_and_clean(filepath)

        print("2/7 -- Handling missing values --")
        df = self.handle_missing_values(df)

        print("3/7 -- Encoding symptoms --")
        df, symptom_list = self.encode_symptoms(df)

        print("4/7 -- Encoding categorical variables --")
        df = self.encode_categorical(df)

        print("5/7 -- Feature engineering --")
        df = self.feature_engineering(df)

        print("6/7 -- Scaling features --")
        df = self.scale_features(df)

        print("7/7 -- Preparing for model input --")
        X, y, feature_cols = self.prepare_for_model(df)

        print("\nPreprocessing pipeline completed successfully.")

        return X, y, feature_cols, df
    
#-----------------------------
# Script execution
#-----------------------------

if __name__ == "__main__":

    #Getting directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)

    #Defining relative paths
    data_raw_dir = os.path.join(backend_dir, 'data', 'raw')
    data_processed_dir = os.path.join(backend_dir, 'data', 'processed')

    #Path to csv file
    csv_file = os.path.join(data_raw_dir, 'respiratory_symptoms_and_treatment.csv')

    #Creating output directory if not done
    os.makedirs(data_processed_dir, exist_ok=True)

    #Initializing preprocessor
    preprocessor = Preprocessor()

    #Running pipeline
    print("=" * 60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    X, y, feature_cols, processed_df = preprocessor.pipeline(csv_file)

    #Printing summary of processed data
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols[:10]}...")  # Show first 10
    print(f"Target distribution:\n{y.value_counts()}")


    #Saving processed data
    print("\nSaving processed data...")
    output_csv = os.path.join(data_processed_dir, 'respiratory_processed.csv')
    processed_df.to_csv(output_csv, index=False)
    print("Processed data saved successfully.")

    #Saving feature names
    print("\nSaving feature names...")
    output_features = os.path.join(data_processed_dir, 'feature_cols.pkl')
    joblib.dump(feature_cols, output_features)
    print("Feature names saved successfully.")

    #Saving preprocessor object
    print("\nSaving preprocessor object...")
    output_preprocessor = os.path.join(data_processed_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, output_preprocessor)
    print("Preprocessor object saved successfully.")

    print("\n" + "=" * 60)
    print("All preprocessing artifacts saved successfully.")
    print("=" * 60)