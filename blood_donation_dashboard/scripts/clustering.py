"""
Clustering Module for Blood Donation Dashboard

This module provides donor profiling through clustering techniques.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data_for_clustering(df, feature_cols=None):
    """
    Prepare data for clustering by selecting relevant features
    and preprocessing them appropriately.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with donor information
    feature_cols : list, optional
        List of feature column names to use for clustering.
        If None, will use default features.
        
    Returns:
    --------
    tuple
        (X_scaled, feature_names) where X_scaled is the prepared feature matrix
    """
    if df is None or df.empty:
        return None, []
    
    # If specific feature columns are provided, use those
    if feature_cols is not None and len(feature_cols) > 0:
        # Check if the specified columns exist in the dataframe
        valid_cols = [col for col in feature_cols if col in df.columns]
        
        if not valid_cols:
            print("None of the specified feature columns exist in the dataframe")
            return None, []
            
        # Separate numeric and categorical features
        numeric_features = df[valid_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df[valid_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create feature dataframe
        X = df[valid_cols].copy()
        
        # Handle missing values
        for col in numeric_features:
            X[col].fillna(X[col].median(), inplace=True)
            
        for col in categorical_features:
            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown", inplace=True)
            
        # Create preprocessing pipeline
        transformers = []
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
            
        if not transformers:
            print("No valid features for clustering")
            return None, []
            
        # Apply preprocessing
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        X_processed = preprocessor.fit_transform(X)
        
        # If categorical features were one-hot encoded, get the feature names
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                # For numeric features, just use the column names
                feature_names.extend(columns)
            elif name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                # For categorical features, get the one-hot encoded feature names
                for col in columns:
                    for cat in transformer.categories_[columns.index(col)]:
                        feature_names.append(f"{col}_{cat}")
        
        return X_processed, feature_names
    
    # Default feature selection if no specific columns are provided
    # Identify potential numeric features
    numeric_features = []
    if 'Age' in df.columns:
        numeric_features.append('Age')
        
    if 'Health_Risk_Score' in df.columns:
        numeric_features.append('Health_Risk_Score')
    
    # Health columns
    health_cols = ['Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
                  'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                  'Cardiaque', 'Tatoué', 'Scarifié']
    
    health_cols_present = [col for col in health_cols if col in df.columns]
    
    # Basic donor info
    donor_info_cols = []
    if 'Previous_Donation' in df.columns:
        donor_info_cols.append('Previous_Donation')
        
    # Categorical features for conversion
    categorical_features = []
    if 'Gender' in df.columns:
        categorical_features.append('Gender')
        
    if 'Location' in df.columns:
        categorical_features.append('Location')
        
    if 'Blood_Type' in df.columns:
        categorical_features.append('Blood_Type')
        
    if 'Age_Group' in df.columns:
        categorical_features.append('Age_Group')
    
    # Check if we have enough features
    all_features = numeric_features + health_cols_present + donor_info_cols + categorical_features
    if not all_features:
        print("Insufficient features for clustering")
        return None, []
        
    # Create feature DataFrame
    X = df[all_features].copy()
    
    # Handle missing values - simple imputation
    for col in numeric_features:
        X[col].fillna(X[col].median(), inplace=True)
        
    for col in categorical_features + donor_info_cols:
        X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown", inplace=True)
    
    for col in health_cols_present:
        X[col].fillna(0, inplace=True)  # Default to no health condition
    
    # Create preprocessing pipeline
    transformers = []
    
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
        
    if health_cols_present:
        transformers.append(('health', StandardScaler(), health_cols_present))
        
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    
    if donor_info_cols:  # These might be binary or categorical
        transformers.append(('donor', OneHotEncoder(handle_unknown='ignore'), donor_info_cols))
    
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    X_processed = preprocessor.fit_transform(X)
    
    # Create meaningful feature names for the transformed features
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name in ['num', 'health']:  # Numeric features - just use column names
            feature_names.extend(columns)
        elif hasattr(transformer, 'get_feature_names_out'):  # One-hot encoded features
            for col in columns:
                for cat in transformer.categories_[columns.index(col)]:
                    feature_names.append(f"{col}_{cat}")
    
    return X_processed, feature_names

def find_optimal_clusters(X, feature_names=None, max_clusters=10):
    """
    Find the optimal number of clusters using silhouette score.
    
    Parameters:
    -----------
    X : array-like
        Prepared feature matrix
    feature_names : list
        List of feature names
    max_clusters : int
        Maximum number of clusters to try
        
    Returns:
    --------
    int
        Optimal number of clusters
    """
    if X is None or X.shape[0] == 0:  # Use shape[0] instead of len(X) for sparse matrices
        return 3  # Default
    
    # Calculate silhouette scores for different cluster numbers
    silhouette_scores = []
    # Ensure we don't have too many clusters - use shape[0] for sparse matrices
    max_samples = X.shape[0]
    cluster_range = range(2, min(max_clusters + 1, max_samples // 5 + 1))
    
    try:
        # Calculate silhouette score for each number of clusters
        for n_clusters in cluster_range:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(X)
            
            if len(set(cluster_labels)) > 1:  # Ensure we have at least 2 clusters
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        
        # Find the optimal number of clusters
        if silhouette_scores and max(silhouette_scores) > 0:
            optimal_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
        else:
            optimal_clusters = min(3, max_samples // 10 + 1)  # Adaptive default
    
    except Exception as e:
        print(f"Error finding optimal clusters: {e}")
        # Adaptive default based on dataset size
        optimal_clusters = min(3, max_samples // 10 + 1)
    
    return optimal_clusters

def perform_clustering(df, n_clusters=None, feature_names=None):
    """
    Perform clustering to identify donor profiles.
    
    Parameters:
    -----------
    df : pd.DataFrame or np.ndarray
        Cleaned dataframe with donor information or preprocessed feature matrix
    n_clusters : int
        Number of clusters to use (if None, will find optimal)
    feature_names : list
        List of feature names for the preprocessed data (if df is a feature matrix)
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with cluster assignments and PCA components
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty) or (isinstance(df, np.ndarray) and df.size == 0):
        return None
    
    # Check if df is already a preprocessed feature matrix
    if not isinstance(df, pd.DataFrame) or (isinstance(df, np.ndarray) and feature_names is not None):
        # If df is a feature matrix, use it directly
        X = df
        # We need the original dataframe to add cluster labels
        # Create a minimal dataframe with just the indices
        result_df = pd.DataFrame(index=range(X.shape[0]))
    else:
        # Prepare data for clustering
        X, feature_names = prepare_data_for_clustering(df)
        result_df = df.copy()
    
    if X is None or X.shape[0] == 0:
        return None
    
    try:
        # Find optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = find_optimal_clusters(X, feature_names)
        
        # Create K-means clustering pipeline
        cluster_pipeline = Pipeline([
            ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
        ])
        
        # Fit pipeline and get cluster labels
        cluster_labels = cluster_pipeline.fit_predict(X)
        
        # Add cluster labels to result dataframe
        result_df['Cluster'] = cluster_labels
        
        # Get PCA components for visualization
        pca = PCA(n_components=min(3, X.shape[1]))
        pca_components = pca.fit_transform(X)
        
        # Store the PCA components for visualization
        result_df.pca_result = pca_components
        
        # Add PCA components to result
        for i in range(min(3, pca_components.shape[1])):  # Add up to 3 components
            result_df[f'PCA_{i+1}'] = pca_components[:, i]
        
        # Create cluster profiles
        profiles = []
        
        # Calculate mean/mode of features by cluster
        for cluster_id in range(n_clusters):
            cluster_row = {'Cluster': cluster_id}
            
            # Get cluster data
            cluster_data = result_df[result_df['Cluster'] == cluster_id]
            
            # Skip if no data in this cluster
            if len(cluster_data) == 0:
                continue
            
            # For each feature, calculate mean or mode
            if isinstance(df, pd.DataFrame):
                # If we have the original dataframe, we can calculate meaningful profiles
                for feature in df.columns:
                    if feature in ['Cluster', 'PCA_1', 'PCA_2', 'PCA_3']:
                        continue
                        
                    # Check data type
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        # For numeric features, calculate mean
                        cluster_row[feature] = cluster_data[feature].mean()
                    else:
                        # For categorical features, calculate mode
                        if not cluster_data[feature].mode().empty:
                            cluster_row[feature] = cluster_data[feature].mode()[0]
            
            profiles.append(cluster_row)
        
        # Add profiles to result dataframe as an attribute
        if profiles:
            result_df.cluster_profiles = pd.DataFrame(profiles)
        
        return result_df
    
    except Exception as e:
        print(f"Error performing clustering: {e}")
        return None

def analyze_clusters(cluster_result, df=None):
    """
    Analyze the clusters to identify key characteristics.
    
    Parameters:
    -----------
    cluster_result : pd.DataFrame
        Result from perform_clustering with cluster assignments
    df : pd.DataFrame, optional
        Original dataframe with additional columns for analysis
        
    Returns:
    --------
    dict
        Dictionary with cluster analysis results
    """
    if cluster_result is None:
        return None
    
    # Initialize analysis dictionary
    analysis = {}
    
    # Get the dataframe with clusters
    df_with_clusters = cluster_result
    
    # Only merge if both dataframes are provided and different
    if df is not None and isinstance(df, pd.DataFrame) and not df.equals(cluster_result):
        # Merge cluster labels back to main dataframe if needed
        df_with_clusters = df.copy()
        try:
            common_index = df_with_clusters.index.intersection(cluster_result.index)
            df_with_clusters.loc[common_index, 'Cluster'] = cluster_result.loc[common_index, 'Cluster']
        except Exception as e:
            print(f"Error merging cluster data: {e}")
            # If merging fails, just use the cluster_result dataframe
            df_with_clusters = cluster_result
    
    # Calculate cluster sizes
    cluster_sizes = df_with_clusters['Cluster'].value_counts().to_dict()
    analysis['cluster_sizes'] = cluster_sizes
    
    # Analyze eligibility by cluster if present
    if 'Eligibility' in df_with_clusters.columns:
        try:
            eligibility_by_cluster = pd.crosstab(
                df_with_clusters['Cluster'], 
                df_with_clusters['Eligibility'],
                normalize='index'
            ) * 100
            
            if 'Eligible' in eligibility_by_cluster.columns:
                analysis['eligibility_by_cluster'] = eligibility_by_cluster
        except Exception as e:
            print(f"Error calculating eligibility by cluster: {e}")
    
    # Analyze previous donation by cluster if present
    if 'Previous_Donation' in df_with_clusters.columns:
        try:
            prev_donation_by_cluster = pd.crosstab(
                df_with_clusters['Cluster'], 
                df_with_clusters['Previous_Donation'],
                normalize='index'
            ) * 100
            
            if 'Yes' in prev_donation_by_cluster.columns:
                analysis['prev_donation_by_cluster'] = prev_donation_by_cluster
        except Exception as e:
            print(f"Error calculating previous donation by cluster: {e}")
    
    # Analyze each cluster
    for cluster in df_with_clusters['Cluster'].unique():
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
        cluster_profile = {}
        
        # Calculate basic stats for numeric columns
        for col in df_with_clusters.select_dtypes(include=['number']).columns:
            if col == 'Cluster' or col.startswith('PCA_'):
                continue
                
            try:
                cluster_profile[f"{col}_mean"] = cluster_data[col].mean()
                cluster_profile[f"{col}_median"] = cluster_data[col].median()
            except Exception:
                pass
        
        # Calculate proportions for categorical columns
        for col in df_with_clusters.select_dtypes(exclude=['number']).columns:
            if col == 'Cluster':
                continue
                
            try:
                value_counts = cluster_data[col].value_counts(normalize=True) * 100
                for value, count in value_counts.items():
                    cluster_profile[f"{col}_{value}"] = count
            except Exception:
                pass
        
        analysis[cluster] = cluster_profile
    
    return analysis

if __name__ == "__main__":
    # Import required modules for testing
    import sys
    sys.path.append('..')
    from scripts.data_processing import load_data, clean_data, generate_features
    
    # Test clustering
    df = load_data()
    if df is not None:
        df_clean = clean_data(df)
        df_features = generate_features(df_clean)
        
        cluster_result = perform_clustering(df_features)
        if cluster_result is not None:
            analysis = analyze_clusters(cluster_result, df_features)
            
            if analysis is not None:
                print("\nCluster Descriptions:")
                for desc in analysis['cluster_descriptions']:
                    print(f"- {desc}")
            
            print("\nClustering completed successfully!")
