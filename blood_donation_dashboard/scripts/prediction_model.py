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
    # Normalize model type naming
    model_type = model_type.lower()
    if model_type in ['rf', 'random_forest', 'randomforest']:
        model_type = 'random_forest'
    elif model_type in ['gb', 'gradient_boosting', 'gradientboosting']:
        model_type = 'gradient_boosting'
    elif model_type in ['lr', 'logistic_regression', 'logisticregression']:
        model_type = 'logistic_regression'
    elif model_type in ['svm', 'support_vector_machine', 'supportvectormachine']:
        model_type = 'svm'
    else:
        # Default to random forest if unrecognized
        model_type = 'random_forest'
    
    try:
        print(f"Training eligibility model using {model_type}")
        
        # Prepare the data
        X, y, features = prepare_data_for_modeling(df, selected_features)
        
        if X is None or y is None:
            print("Failed to prepare data for modeling")
            return None
            
        # Print the features used
        print(f"Using features: {features}")
        
        # Ensure special preprocessing for health condition features
        # Convert them to binary (0/1) values
        health_conditions = ['Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré', 
                             'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                             'Cardiaque', 'Tatoué', 'Scarifié']
        
        for feature in health_conditions:
            if feature in X.columns:
                # Convert any non-zero value to 1
                X[feature] = X[feature].apply(lambda x: 1 if x in [1, '1', 'Yes', 'Oui', True] else 0)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = [f for f in X.columns if f not in numeric_features]
        
        # Store training data columns for feature alignment during prediction
        training_columns = X.columns.tolist()
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Select model based on type
        if model_type == 'random_forest':
            # Tune RF to prevent overfitting to Health_Risk_Score
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5,  # Limit depth to prevent overfitting
                min_samples_split=5,  # Require more samples to split
                min_samples_leaf=2,
                class_weight='balanced',  # Balance classes
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,  # Lower learning rate
                max_depth=3,  # Shallow trees
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                C=1.0,  # Regularization strength
                class_weight='balanced',  # Balance classes
                solver='liblinear',
                random_state=42
            )
        elif model_type == 'svm':
            from sklearn.svm import SVC
            model = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        else:
            # Default to Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit the model
        print("Fitting model...")
        pipeline.fit(X_train, y_train)
        
        # Track the feature names after transformation
        # Store preprocessed X_train data for reference during prediction
        # This helps ensure the same features are used
        X_train_transformed = preprocessor.transform(X_train)
        
        # Make predictions on test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Extract feature importances if available
        feature_importances = {}
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            try:
                # Extract importances directly from the model
                importances = pipeline.named_steps['model'].feature_importances_
                
                # Create a simple mapping of feature index to importance value
                feature_importances = {f"feature_{i}": imp for i, imp in enumerate(importances)}
                
                # Now try to map the original feature names to importances
                # This is complex because of OneHotEncoding, but we can give our best effort
                if numeric_features:
                    # Match numeric features to their respective importances
                    for i, feature in enumerate(numeric_features):
                        if i < len(importances):
                            feature_importances[feature] = importances[i]
                
            except Exception as e:
                print(f"Error extracting feature importances: {e}")
                # Fallback - just use indices
                importances = pipeline.named_steps['model'].feature_importances_
                feature_importances = {f"feature_{i}": imp for i, imp in enumerate(importances)}
                
        elif hasattr(model, 'coef_'):
            # For linear models like logistic regression
            try:
                # Extract coefficients directly from the model
                coefs = pipeline.named_steps['model'].coef_[0]
                
                # Create a simple mapping of feature index to coefficient value
                feature_importances = {f"feature_{i}": abs(c) for i, c in enumerate(coefs)}
                
                # Now try to map the original feature names to coefficients
                if numeric_features:
                    # Match numeric features to their respective coefficients
                    for i, feature in enumerate(numeric_features):
                        if i < len(coefs):
                            feature_importances[feature] = abs(coefs[i])
                
            except Exception as e:
                print(f"Error extracting feature importances: {e}")
                # Fallback - just use indices
                coefs = pipeline.named_steps['model'].coef_[0]
                feature_importances = {f"feature_{i}": abs(c) for i, c in enumerate(coefs)}
        
        # Print metrics
        print(f"Model metrics: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
        
        # Return model and metrics with additional information for prediction
        return {
            'model': pipeline,
            'metrics': metrics,
            'feature_importances': feature_importances,
            'features': features,
            'X_test': X_test,
            'training_columns': training_columns,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'y_test': y_test
        }
        
    except Exception as e:
        print(f"Error training eligibility model: {e}")
        import traceback
        traceback.print_exc()
        return None

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

def predict_eligibility(model_result, input_data):
    """
    Predict eligibility for new donor data.
    
    Parameters:
    -----------
    model_result : dict
        Dictionary returned by train_eligibility_model
    input_data : dict
        Dictionary with feature values for a new donor
        
    Returns:
    --------
    tuple
        (prediction, probability, prediction_explanation)
    """
    try:
        # Check if model_result is valid
        if not model_result or not isinstance(model_result, dict):
            print("Invalid model result")
            return None, None, "Invalid model"
            
        # Get model, features from result
        pipeline = model_result.get('model')  # Use the entire pipeline
        features = model_result.get('features')
        feature_importances = model_result.get('feature_importances', {})
        
        if not pipeline or not features:
            print("Missing pipeline or features in model result")
            return None, None, "Missing model components"
        
        # Convert input_data to dataframe with correct feature order
        input_df = pd.DataFrame([input_data])
        
        # Preprocess health condition inputs
        health_conditions = ['Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré', 
                            'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                            'Cardiaque', 'Tatoué', 'Scarifié']
        
        # Convert health condition Yes/No or Oui/Non to binary values for prediction
        for feature in health_conditions:
            if feature in input_df.columns:
                # Convert Oui/Yes to 1 and Non/No to 0
                if input_df[feature].iloc[0] in ['Oui', 'Yes']:
                    input_df[feature] = 1
                else:
                    input_df[feature] = 0
        
        # Only select the features that were used for model training
        # This is critical to avoid the feature mismatch error
        available_features = [f for f in features if f in input_df.columns]
        if len(available_features) != len(features):
            print(f"Warning: Only {len(available_features)}/{len(features)} features available")
            missing_features = [f for f in features if f not in input_df.columns]
            print(f"Missing features: {missing_features}")
        
        # Select only the features required by the model
        input_df = input_df[available_features]
        
        # Make prediction using the full pipeline
        # The pipeline will handle the preprocessing automatically
        try:
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]  # Probability of class 1
            
            # Generate explanation based on feature importances
            explanation = generate_prediction_explanation(
                prediction, 
                probability, 
                input_df, 
                feature_importances
            )
            
            return prediction, probability, explanation
        except ValueError as e:
            # If there's still a feature mismatch, try a more direct approach
            print(f"Error in pipeline prediction: {e}")
            print("Attempting alternative prediction approach...")
            
            # If we're at this point, we need to manually apply the preprocessing
            # by getting the training data columns and adapting our input
            if 'X_test' in model_result:
                # Get the column names used during training
                X_test = model_result.get('X_test')
                model_features = X_test.columns.tolist()
                
                # Create a DataFrame with zeros for all model features
                aligned_input = pd.DataFrame(0, index=[0], columns=model_features)
                
                # Fill in the available values from the input data
                for col in input_df.columns:
                    if col in aligned_input.columns:
                        aligned_input[col] = input_df[col].values
                
                # Try prediction again with the aligned input
                model_only = pipeline.named_steps.get('model', pipeline)
                prediction = model_only.predict(aligned_input)[0]
                probability = model_only.predict_proba(aligned_input)[0][1]
                
                explanation = "Prediction made with limited feature matching. Results may be less accurate."
                return prediction, probability, explanation
            else:
                return None, None, f"Error: {str(e)}"
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"

def generate_prediction_explanation(prediction, probability, input_data, feature_importances):
    """
    Generate a human-readable explanation for the prediction.
    
    Parameters:
    -----------
    prediction : int
        Predicted class (0 or 1)
    probability : float
        Probability of positive class
    input_data : pd.DataFrame
        Input data used for prediction
    feature_importances : dict
        Feature importances from the model
        
    Returns:
    --------
    str
        Human-readable explanation
    """
    try:
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get top 5 most important features
        top_features = sorted_features[:5]
        
        # Initialize explanation
        if prediction == 1:
            explanation = f"Donor predicted as ELIGIBLE with {probability:.1%} confidence.\n\n"
        else:
            explanation = f"Donor predicted as NOT ELIGIBLE with {(1-probability):.1%} confidence.\n\n"
        
        explanation += "Top factors influencing this prediction:\n"
        
        # Add top features to explanation
        for feature_name, importance in top_features:
            # Clean up feature name if it's from one-hot encoding
            display_name = feature_name
            if '_x0_' in feature_name or '_x1_' in feature_name:
                base_name = feature_name.split('_x')[0]
                value = feature_name.split('_x')[1].split('_')[1]
                display_name = f"{base_name} = {value}"
            
            # Try to get the actual value from input data
            value = None
            # Check if the exact feature name exists in input_data
            if feature_name in input_data.columns:
                value = input_data[feature_name].iloc[0]
            # For one-hot encoded features, check if base name exists
            elif '_x0_' in feature_name or '_x1_' in feature_name:
                base_name = feature_name.split('_x')[0]
                if base_name in input_data.columns:
                    value = input_data[base_name].iloc[0]
            
            if value is not None:
                explanation += f"- {display_name} (value: {value}, importance: {importance:.3f})\n"
            else:
                explanation += f"- {display_name} (importance: {importance:.3f})\n"
        
        explanation += "\nNote: The higher the importance value, the more influence that factor has on the prediction."
        
        return explanation
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return f"Prediction made, but could not generate explanation: {str(e)}"

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
