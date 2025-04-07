"""
Data Processing Module for Blood Donation Dashboard

This module handles data loading, cleaning, and preprocessing functions
for the blood donation dashboard.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

def load_data(filepath='../data/blood_donation_data.csv'):
    """
    Load the blood donation data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing blood donation data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with consistent column names
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Clean and preprocess the blood donation data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw blood donation dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed dataframe
    """
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = [col.strip() for col in df_clean.columns]
    
    # Handle gender data
    gender_mapping = {
        'Homme': 'Male', 'homme': 'Male', 'Male': 'Male', 'male': 'Male', 'M': 'Male', 'HOMME': 'Male',
        'Femme': 'Female', 'femme': 'Female', 'Female': 'Female', 'female': 'Female', 'F': 'Female', 'FEMME': 'Female'
    }
    
    # Use the appropriate gender column
    if 'Sexe' in df_clean.columns:
        df_clean['Gender'] = df_clean['Sexe'].map(gender_mapping)
    elif 'SEXE' in df_clean.columns:
        df_clean['Gender'] = df_clean['SEXE'].map(gender_mapping)
    elif 'sexe' in df_clean.columns:
        df_clean['Gender'] = df_clean['sexe'].map(gender_mapping)
    elif 'Genre' in df_clean.columns:
        df_clean['Gender'] = df_clean['Genre'].map(gender_mapping)
    
    # Standardize eligibility data
    eligibility_cols = ['ÉLIGIBILITÉ_AU_DON.', 'Eligibilité', 'ELIGIBILITE', 'eligibilite', 'ÉLIGIBILITÉ']
    eligibility_col = next((col for col in eligibility_cols if col in df_clean.columns), None)
    
    if eligibility_col:
        eligible_values = ['Eligible', 'eligible', 'OUI', 'Oui', 'oui', 'Yes', 'yes']
        df_clean['Eligibility'] = df_clean[eligibility_col].apply(
            lambda x: 'Eligible' if str(x).strip() in eligible_values else 'Ineligible'
        )
    
    # Standardize previous donation data
    prev_donation_cols = ['A-t-il_(elle)_déjà_donné_le_sang', 'A-t-il (elle) déjà donné le sang', 
                          'Déjà donné', 'DEJA_DONNE', 'Déjà_donné']
    prev_donation_col = next((col for col in prev_donation_cols if col in df_clean.columns), None)
    
    if prev_donation_col:
        yes_values = ['Yes', 'YES', 'Oui', 'OUI', 'oui', 'yes']
        no_values = ['No', 'NO', 'Non', 'NON', 'non', 'no']
        df_clean['Previous_Donation'] = df_clean[prev_donation_col].apply(
            lambda x: 'Yes' if str(x).strip() in yes_values
            else ('No' if str(x).strip() in no_values else 'Unknown')
        )
    
    # Handle date fields
    date_cols = ['Date de remplissage de la fiche', 'Date de naissance']
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Create age from date of birth if not present
    if 'Age' not in df_clean.columns and 'Date de naissance' in df_clean.columns:
        current_year = datetime.now().year
        df_clean['Age'] = df_clean['Date de naissance'].apply(
            lambda x: current_year - x.year if pd.notnull(x) else np.nan
        )
    elif 'age' in df_clean.columns:
        df_clean['Age'] = df_clean['age']
    
    # Create standardized health condition columns
    health_cols = ['Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
                  'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                  'Cardiaque', 'Tatoué', 'Scarifié']
    
    for col in health_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: True if str(x).strip().lower() in ['yes', 'oui', 'true'] else False
            )
    
    # Clean location data
    location_cols = ['Arrondissement_de_résidence', 'Arrondissement de résidence', 'Quartier_de_Résidence', 'Quartier de Résidence']
    for col in location_cols:
        if col in df_clean.columns:
            # Extract simplified name for standardization
            new_col_name = 'District' if 'Arrondissement' in col else 'Neighborhood'
            df_clean[new_col_name] = df_clean[col].apply(
                lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).strip() if pd.notnull(x) else 'Unknown'
            )
    
    return df_clean

def generate_features(df):
    """
    Generate additional features for analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned blood donation dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional features
    """
    if df is None:
        return None
    
    df_features = df.copy()
    
    # Create donation season (if date column exists)
    if 'Date de remplissage de la fiche' in df_features.columns:
        df_features['Donation_Month'] = df_features['Date de remplissage de la fiche'].dt.month
        
        # Map months to seasons
        season_mapping = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 
            4: 'Spring', 5: 'Spring', 6: 'Summer',
            7: 'Summer', 8: 'Summer', 9: 'Fall', 
            10: 'Fall', 11: 'Fall', 12: 'Winter'
        }
        df_features['Donation_Season'] = df_features['Donation_Month'].map(season_mapping)
    
    # Age groups
    if 'Age' in df_features.columns:
        bins = [0, 18, 25, 35, 45, 55, 65, 100]
        labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        df_features['Age_Group'] = pd.cut(df_features['Age'], bins=bins, labels=labels, right=False)
    
    # Health risk score (count of health conditions)
    health_cols = ['Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
                  'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                  'Cardiaque', 'Tatoué', 'Scarifié']
    
    health_cols_present = [col for col in health_cols if col in df_features.columns]
    if health_cols_present:
        df_features['Health_Risk_Score'] = df_features[health_cols_present].sum(axis=1)
    
    return df_features

def get_geographical_data(df):
    """
    Extract geographical data for mapping.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned blood donation dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with geographical information for mapping
    """
    if df is None:
        return None
    
    geo_df = df.copy()
    
    # Ensure we have latitude and longitude columns
    if 'latitude' in geo_df.columns and 'longitude' in geo_df.columns:
        geo_df = geo_df[['latitude', 'longitude', 'ID']].drop_duplicates()
        geo_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
        
        # Count donors per location
        location_counts = geo_df.groupby(['lat', 'lon']).size().reset_index(name='donor_count')
        return location_counts
    else:
        print("Warning: No geographical coordinates found in dataset")
        return None

def prepare_data_for_ml(df):
    """
    Prepare data for machine learning models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned blood donation dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe ready for machine learning
    """
    if df is None:
        return None
    
    ml_df = df.copy()
    
    # Define features and target
    ml_features = []
    
    # Demographic features
    if 'Age' in ml_df.columns:
        ml_features.append('Age')
    
    # Gender (encode as numeric)
    if 'Gender' in ml_df.columns:
        ml_df['Gender_Encoded'] = ml_df['Gender'].map({'Male': 1, 'Female': 0})
        ml_features.append('Gender_Encoded')
    
    # Health conditions
    health_cols = ['Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
                  'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                  'Cardiaque', 'Tatoué', 'Scarifié']
    
    for col in health_cols:
        if col in ml_df.columns:
            ml_df[f"{col}_Encoded"] = ml_df[col].map({True: 1, False: 0})
            ml_features.append(f"{col}_Encoded")
    
    # Previous donation (encode as numeric)
    if 'Previous_Donation' in ml_df.columns:
        ml_df['Previous_Donation_Encoded'] = ml_df['Previous_Donation'].map({'Yes': 1, 'No': 0})
        ml_features.append('Previous_Donation_Encoded')
    
    # Target variable (eligibility)
    target = None
    if 'Eligibility' in ml_df.columns:
        ml_df['Eligibility_Encoded'] = ml_df['Eligibility'].map({'Eligible': 1, 'Ineligible': 0})
        target = 'Eligibility_Encoded'
    
    if not ml_features or target is None:
        print("Warning: Insufficient features or no target variable for ML modeling")
        return None
    
    # Return features and target
    return ml_df[ml_features + [target]]

if __name__ == "__main__":
    # Test data processing
    df = load_data()
    if df is not None:
        df_clean = clean_data(df)
        df_features = generate_features(df_clean)
        geo_data = get_geographical_data(df_features)
        ml_data = prepare_data_for_ml(df_features)
        
        print("Data processing completed successfully!")
