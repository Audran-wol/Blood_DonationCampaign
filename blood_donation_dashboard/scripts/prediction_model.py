"""
Prediction Model Module for Blood Donation Dashboard

This module provides machine learning models to predict donor eligibility.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def prepare_data_for_modeling(df, selected_features=None):
    """
    Prepare data for machine learning modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with donor information
    selected_features : list, optional
        Specific features to use. If None, will use all viable features
        
    Returns:
    --------
    tuple
        (X, y, feature_names) or (None, None, None) if insufficient data
    """
    if df is None or df.empty:
        return None, None, None
    
    # Check if eligibility data is available
    if 'Eligibility' in df.columns:
        y = df['Eligibility'].map({'Eligible': 1, 'Ineligible': 0})
    elif 'target' in df.columns:
        # Use target column if it exists (for the modified train_model function)
        y = df['target']
    else:
        print("No eligibility or target data available for modeling")
        return None, None, None
    
    # Use provided features if available
    if selected_features is not None and len(selected_features) > 0:
        features = [f for f in selected_features if f in df.columns and f not in ['Eligibility', 'target']]
        if len(features) == 0:
            print("None of the selected features are available in the dataframe")
            return None, None, None
    else:
        # Auto-select features if none provided
        # Define potential features
        numeric_features = []
        if 'Age' in df.columns:
            numeric_features.append('Age')
        
        if 'Health_Risk_Score' in df.columns:
            numeric_features.append('Health_Risk_Score')
        
        # Health columns - add direct health indicators
        health_cols = ['Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
                      'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                      'Cardiaque', 'Tatoué', 'Scarifié']
        
        health_cols_present = [col for col in health_cols if col in df.columns]
        numeric_features.extend(health_cols_present)
        
        # Categorical features
        categorical_features = []
        
        if 'Gender' in df.columns:
            categorical_features.append('Gender')
        
        if 'Previous_Donation' in df.columns:
            categorical_features.append('Previous_Donation')
        
        if 'Age_Group' in df.columns:
            categorical_features.append('Age_Group')
        
        # Combine all features
        features = numeric_features + categorical_features
    
    # Check if we have enough features
    if len(features) == 0:
        print("Insufficient features for modeling")
        return None, None, None
    
    # Create feature dataframe
    X = df[features].copy()
    
    # Handle missing values - simple imputation
    for col in X.select_dtypes(include=['int64', 'float64']).columns:
        X[col].fillna(X[col].median(), inplace=True)
    
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown", inplace=True)
    
    # Check if we have enough data after cleaning
    if len(X) < 10:  # reduced minimum
        print("Insufficient data for modeling after cleaning")
        return None, None, None
    
    return X, y, features

def train_eligibility_model(df, model_type='rf', test_size=0.2, selected_features=None):
    """
    Train a model to predict donor eligibility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with donor information
    model_type : str
        Type of model to train ('rf' for RandomForest, 'gb' for GradientBoosting, 'lr' for LogisticRegression)
    test_size : float
        Proportion of data to use for testing
    selected_features : list, optional
        List of specific features to use for training
        
    Returns:
    --------
    dict
        Dictionary with model, pipeline, metrics, and feature importances
    """
    # Print information for debugging
    print(f"Starting model training with {model_type} model")
    print(f"Dataframe shape: {df.shape}")
    if selected_features:
        print(f"Selected features: {selected_features}")
    
    # Prepare data
    X, y, features = prepare_data_for_modeling(df, selected_features)
    
    if X is None or y is None:
        print("Data preparation failed, no features or target available")
        return None
    
    print(f"After preparation: X shape={X.shape}, features={features}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Create preprocessing pipeline
    transformers = []
    
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    
    if not transformers:
        print("No transformers could be created. Check your feature types.")
        return None
    
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    # Select and configure model
    if model_type == 'random_forest' or model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
    elif model_type == 'gradient_boosting' or model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    elif model_type == 'logistic_regression' or model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'saga']
        }
    else:  # default to logistic regression
        print(f"Unknown model type: {model_type}, defaulting to Logistic Regression")
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'saga']
        }
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Perform grid search if there's enough data
    if len(X_train) >= 100 and len(X_train) > 10 * len(features):
        try:
            print("Performing grid search...")
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            print(f"Grid search complete. Best params: {grid_search.best_params_}")
        except Exception as e:
            print(f"Grid search failed: {e}. Falling back to default model.")
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
    else:
        # Just fit with default parameters if data is limited
        print("Using default parameters (data is limited or high-dimensional)")
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
    
    # Evaluate model
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Get feature importances if applicable
    feature_importances = {}
    
    try:
        if model_type in ['rf', 'gb', 'random_forest', 'gradient_boosting']:
            # Extract feature names after one-hot encoding
            final_feature_names = []
            
            # Get numeric features directly
            if numeric_features:
                final_feature_names.extend(numeric_features)
            
            # Get transformed categorical feature names
            if categorical_features:
                try:
                    for i, (name, transformer, column) in enumerate(preprocessor.transformers_):
                        if name == 'cat':
                            cat_feature_names = transformer.get_feature_names_out(input_features=transformer.feature_names_in_)
                            final_feature_names.extend(cat_feature_names)
                except Exception as e:
                    print(f"Error getting categorical feature names: {e}")
            
            # Get importances
            importances = best_pipeline.named_steps['classifier'].feature_importances_
            
            # Check lengths match
            if len(importances) == len(final_feature_names):
                for i, importance in enumerate(importances):
                    feature_importances[final_feature_names[i]] = importance
            else:
                print(f"Warning: Feature importances length ({len(importances)}) doesn't match feature names length ({len(final_feature_names)})")
                # Just use index as feature name
                for i, importance in enumerate(importances):
                    feature_importances[f"feature_{i}"] = importance
        
        elif model_type in ['lr', 'logistic_regression']:
            # Similar approach but with coefficients instead of feature_importances_
            final_feature_names = []
            
            # Get numeric features directly
            if numeric_features:
                final_feature_names.extend(numeric_features)
            
            # Get transformed categorical feature names
            if categorical_features:
                try:
                    for i, (name, transformer, column) in enumerate(preprocessor.transformers_):
                        if name == 'cat':
                            cat_feature_names = transformer.get_feature_names_out(input_features=transformer.feature_names_in_)
                            final_feature_names.extend(cat_feature_names)
                except Exception as e:
                    print(f"Error getting categorical feature names: {e}")
            
            # Get coefficients
            coefficients = best_pipeline.named_steps['classifier'].coef_[0]
            
            # Check lengths match
            if len(coefficients) == len(final_feature_names):
                for i, coef in enumerate(coefficients):
                    feature_importances[final_feature_names[i]] = abs(coef)  # Use absolute value
            else:
                print(f"Warning: Coefficients length ({len(coefficients)}) doesn't match feature names length ({len(final_feature_names)})")
                # Just use index as feature name
                for i, coef in enumerate(coefficients):
                    feature_importances[f"feature_{i}"] = abs(coef)
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
    
    # Print results summary
    print(f"Model training complete. Metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Return results
    return {
        'model': best_pipeline,
        'metrics': metrics,
        'feature_importances': feature_importances,
        'features': features,
        'test_data': {'X': X_test, 'y': y_test}
    }

def train_model(data, model_type='rf', test_size=0.2, target=None, selected_features=None):
    """
    Train a model to predict donor eligibility.
    
    Flexible function that can handle both DataFrame with target column
    and separate X, y inputs.
    
    Parameters:
    -----------
    data : pd.DataFrame or np.ndarray
        Either a complete dataframe with target column, or feature matrix X
    model_type : str
        Type of model to train (rf, random_forest, gb, gradient_boosting, lr, logistic_regression)
    test_size : float
        Proportion of data to use for testing
    target : pd.Series, optional
        Target values if data is feature matrix X
    selected_features : list, optional
        List of specific features to use for training
        
    Returns:
    --------
    dict
        Dictionary with model, pipeline, metrics, and feature importances
    """
    # Check if data is X and second argument is y
    if isinstance(data, (pd.DataFrame, np.ndarray)) and isinstance(model_type, (pd.Series, np.ndarray)):
        # If model_type is actually y, adjust parameters
        X = data
        y = model_type
        model_type = test_size if isinstance(test_size, str) else 'rf'
        # Create a temporary dataframe
        df = pd.DataFrame(X)
        df['target'] = y
        return train_eligibility_model(df, model_type=model_type, selected_features=selected_features)
    
    # Original behavior
    return train_eligibility_model(data, model_type=model_type, test_size=test_size, selected_features=selected_features)

def evaluate_model(model_result, X_test=None, y_test=None):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model_result : dict
        Dictionary with trained model and pipeline
    X_test : pd.DataFrame, optional
        Test feature data. If None, uses test data from model_result
    y_test : pd.Series, optional
        Test target data. If None, uses test data from model_result
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    if model_result is None:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'auc': 0
        }
    
    # Use the model's pipeline if available, otherwise use the model directly
    if 'pipeline' in model_result:
        pipeline = model_result['pipeline']
    elif 'model' in model_result:
        pipeline = model_result['model']
    else:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'auc': 0
        }
    
    # Use provided test data if available, otherwise use test data from model_result
    if X_test is None or y_test is None:
        if 'test_data' in model_result:
            X_test = model_result['test_data']['X']
            y_test = model_result['test_data']['y']
        else:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'auc': 0
            }
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    # Add AUC if probability predictions are available
    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            metrics['auc'] = auc(fpr, tpr)
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            metrics['auc'] = 0
    else:
        metrics['auc'] = 0
    
    return metrics

def save_model(model_result, filepath='../models/eligibility_model.pkl'):
    """
    Save the trained model to a file.
    
    Parameters:
    -----------
    model_result : dict
        Dictionary with model and associated data
    filepath : str
        Path to save the model
        
    Returns:
    --------
    bool
        True if saved successfully, False otherwise
    """
    if model_result is None or 'model' not in model_result:
        return False
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model_result['model'], f)
        
        print(f"Model saved to {filepath}")
        return True
    
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(filepath='../models/eligibility_model.pkl'):
    """
    Load a trained model from a file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    object
        Loaded model or None if loading failed
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_eligibility(model, input_data):
    """
    Predict eligibility for new donor data.
    
    Parameters:
    -----------
    model : object
        Trained model pipeline
    input_data : pd.DataFrame
        Input data for prediction
        
    Returns:
    --------
    tuple
        (predictions, probabilities) or (None, None) if prediction failed
    """
    if model is None or input_data is None or input_data.empty:
        print("Model or input data is None or empty")
        return None, None
    
    try:
        # Print debugging information
        print(f"Input data shape: {input_data.shape}")
        print(f"Input data columns: {input_data.columns.tolist()}")
        print(f"Input data types: {input_data.dtypes}")
        
        # Check if model pipeline has a feature list we should verify against
        required_features = model.get('features', None)
        if required_features is not None:
            # If model has a features list, ensure input data matches
            missing_features = [f for f in required_features if f not in input_data.columns]
            if missing_features:
                print(f"Missing required features: {missing_features}")
                return None, None
        
        # Get the actual model pipeline
        model_pipeline = model.get('model', model)
        
        # Make predictions using the model pipeline
        print("Making prediction...")
        predictions = model_pipeline.predict(input_data)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model_pipeline, 'predict_proba'):
            probabilities = model_pipeline.predict_proba(input_data)
        
        print(f"Prediction successful. Result: {predictions}")
        return predictions, probabilities
    
    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_model_performance(model_result):
    """
    Create visualizations of model performance.
    
    Parameters:
    -----------
    model_result : dict
        Dictionary with model results
        
    Returns:
    --------
    dict
        Dictionary of plotly figures
    """
    if model_result is None:
        return {}
    
    figures = {}
    
    # Confusion matrix
    if 'metrics' in model_result and 'confusion_matrix' in model_result['metrics']:
        cm = model_result['metrics']['confusion_matrix']
        
        # Create confusion matrix heatmap
        cm_fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Ineligible', 'Eligible'],
            y=['Ineligible', 'Eligible'],
            text_auto=True,
            title="Confusion Matrix",
            color_continuous_scale='Blues'
        )
        figures['confusion_matrix'] = cm_fig
    
    # Feature importance
    if 'feature_importances' in model_result and model_result['feature_importances']:
        # Convert to dataframe for plotting
        fi_df = pd.DataFrame({
            'Feature': list(model_result['feature_importances'].keys()),
            'Importance': list(model_result['feature_importances'].values())
        }).sort_values('Importance', ascending=False)
        
        # Take top 15 features to avoid overcrowding
        fi_df = fi_df.head(15)
        
        # Create feature importance bar chart
        fi_fig = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            labels={'Importance': 'Importance', 'Feature': 'Feature'},
            color='Importance',
            color_continuous_scale='Viridis'
        )
        figures['feature_importance'] = fi_fig
    
    # ROC curve
    if ('test_data' in model_result and 
        'X' in model_result['test_data'] and 
        'y' in model_result['test_data'] and
        'model' in model_result):
        
        X_test = model_result['test_data']['X']
        y_test = model_result['test_data']['y']
        model = model_result['model']
        
        if hasattr(model, 'predict_proba'):
            try:
                y_scores = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                # Create ROC curve
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'ROC curve (area = {roc_auc:.2f})',
                    mode='lines',
                    line=dict(color='darkorange', width=2)
                ))
                
                roc_fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random',
                    mode='lines',
                    line=dict(color='navy', width=2, dash='dash')
                ))
                
                roc_fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    xaxis=dict(constrain='domain'),
                    autosize=False,
                    width=700,
                    height=500,
                    legend=dict(
                        x=0.05,
                        y=0.95,
                        traceorder='normal',
                        font=dict(family='sans-serif', size=12, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.5)',
                        bordercolor='rgba(0, 0, 0, 0.5)',
                        borderwidth=1
                    )
                )
                
                figures['roc_curve'] = roc_fig
            except Exception as e:
                print(f"Error creating ROC curve: {e}")
    
    return figures

if __name__ == "__main__":
    # Import required modules for testing
    import sys
    sys.path.append('..')
    from scripts.data_processing import load_data, clean_data, generate_features
    
    # Test model training
    df = load_data()
    if df is not None:
        df_clean = clean_data(df)
        df_features = generate_features(df_clean)
        
        # Train model
        print("Training Random Forest model...")
        rf_result = train_eligibility_model(df_features, model_type='rf')
        
        if rf_result is not None:
            print(f"Model metrics: Accuracy={rf_result['metrics']['accuracy']:.2f}, "
                  f"F1 Score={rf_result['metrics']['f1']:.2f}")
            
            # Save model
            save_model(rf_result)
            
            print("Model training and evaluation completed successfully!")
