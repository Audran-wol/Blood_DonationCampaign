"""
Eligibility Prediction Page

This module provides functionality for predicting donor eligibility based on various factors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import from scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from scripts.prediction_model import train_model, evaluate_model, predict_eligibility
from scripts.visualization import create_prediction_vis

def show_eligibility_prediction(df):
    """
    Display the eligibility prediction page.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Filtered dataframe with donor information
    """
    # Header
    st.markdown('<div class="main-header">Eligibility Prediction</div>', unsafe_allow_html=True)
    st.markdown("Predict donor eligibility using machine learning models.")
    st.markdown("---")
    
    # Check if eligibility data is available
    if 'Eligibility' not in df.columns:
        st.error("Eligibility data is required for prediction modeling.")
        return
    
    # Create tabs for different aspects of prediction
    tab1, tab2 = st.tabs(["Model Performance", "Eligibility Predictor"])
    
    with tab1:
        st.markdown('<div class="sub-header">Model Performance Analysis</div>', unsafe_allow_html=True)
        
        # Model selection
        model_options = ["Random Forest", "Logistic Regression", "Support Vector Machine", "Gradient Boosting"]
        selected_model = st.selectbox("Select Model Type", model_options)
        
        # Feature selection
        feature_cols = [col for col in df.columns if col not in ['Eligibility', 'target'] and pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if col not in ['Eligibility', 'target'] and not pd.api.types.is_numeric_dtype(df[col])]
        
        st.markdown("### Feature Selection")
        
        # Group features by category
        feature_categories = {
            "Demographics": ["Age", "Gender", "Age_Group", "Blood_Type", "Weight"],
            "Health Indicators": ["Health_Risk_Score", "Antécédent_de_transfusion", "Porteur(HIV,hbs,hcv)", 
                                 "Opéré", "Drepanocytaire", "Diabétique", "Hypertendus", 
                                 "Asthmatiques", "Cardiaque", "Tatoué", "Scarifié"],
            "Donation History": ["Previous_Donation", "Frequency", "Last_Donation_Date"],
            "Geographic": ["Region", "City", "Distance_to_Center"]
        }
        
        # Get available features from dataframe
        available_features = {}
        for category, features in feature_categories.items():
            available_in_category = [f for f in features if f in df.columns]
            if available_in_category:
                available_features[category] = available_in_category
        
        # Allow feature category selection
        selected_categories = st.multiselect(
            "Select Feature Categories", 
            options=list(available_features.keys()),
            default=list(available_features.keys())[:2] if available_features else []
        )
        
        # Compile selected features from categories
        selected_features = []
        for category in selected_categories:
            selected_features.extend(available_features.get(category, []))
        
        # Add option to manually select additional numeric features not in categories
        other_numeric = [col for col in feature_cols if col not in [f for cat_features in available_features.values() for f in cat_features]]
        if other_numeric:
            with st.expander("Additional Numeric Features"):
                selected_other_numeric = st.multiselect(
                    "Select Additional Features",
                    options=other_numeric,
                    default=other_numeric[:min(3, len(other_numeric))]
                )
                selected_features.extend(selected_other_numeric)
        
        # Add option for categorical features
        if categorical_cols:
            with st.expander("Categorical Features"):
                selected_categorical = st.multiselect(
                    "Select Categorical Features",
                    options=categorical_cols,
                    default=[]
                )
                selected_features.extend(selected_categorical)
        
        # Check if we have enough features
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features for model training.")
        else:
            # Display selected features
            st.write(f"Selected {len(selected_features)} features for model training")
            
            # Training parameters
            col1, col2, col3 = st.columns(3)
            
            test_size = col1.slider("Test Size (%)", 10, 40, 20) / 100
            
            # Train model on button click
            if st.button("Train and Evaluate Model"):
                with st.spinner("Training model..."):
                    try:
                        # Map model selection to actual model type
                        model_mapping = {
                            "Random Forest": "random_forest",
                            "Logistic Regression": "logistic_regression", 
                            "Support Vector Machine": "svm",
                            "Gradient Boosting": "gradient_boosting"
                        }
                        model_name = model_mapping.get(selected_model, "random_forest")
                        
                        # Create a temporary DataFrame with only the selected features and target
                        model_df = df[selected_features].copy()
                        model_df['target'] = df['Eligibility'].map({'Eligible': 1, 'Ineligible': 0})
                        
                        st.info(f"Training {selected_model} model with {len(selected_features)} features")
                        
                        # Train the model with the selected features
                        model = train_model(
                            data=model_df, 
                            model_type=model_name,
                            test_size=test_size,
                            selected_features=selected_features
                        )
                        
                        if model is not None:
                            # Evaluate model
                            evaluation_results = model['metrics']
                            
                            # Display performance metrics
                            st.markdown("### Model Performance Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("Accuracy", f"{evaluation_results['accuracy']*100:.1f}%")
                            col2.metric("Precision", f"{evaluation_results['precision']*100:.1f}%")
                            col3.metric("Recall", f"{evaluation_results['recall']*100:.1f}%")
                            col4.metric("F1 Score", f"{evaluation_results['f1']*100:.1f}%")
                            
                            # Feature importance
                            if 'feature_importances' in model and model['feature_importances']:
                                st.markdown("### Feature Importance")
                                feature_imp = pd.DataFrame({
                                    'Feature': list(model['feature_importances'].keys()),
                                    'Importance': list(model['feature_importances'].values())
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    feature_imp, 
                                    x='Importance', 
                                    y='Feature',
                                    orientation='h',
                                    title='Feature Importance',
                                    color='Importance',
                                    color_continuous_scale='Viridis'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Store model for prediction
                            st.session_state.model = model
                            st.session_state.selected_features = selected_features
                            
                            st.success("Model trained successfully! You can now make predictions.")
                        else:
                            st.error("Model training failed. Check the feature selection.")
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        st.info("Try selecting different features or a different model type.")
    
    with tab2:
        st.markdown('<div class="sub-header">Predict Donor Eligibility</div>', unsafe_allow_html=True)
        
        # Check if model is available
        if 'model' not in st.session_state:
            st.info("Please train a model in the 'Model Performance' tab before making predictions.")
        else:
            # Get model info
            model = st.session_state.model
            features = st.session_state.selected_features
            
            # Create input form for prediction
            st.markdown("### Enter Donor Information")
            
            # Create input fields for each feature
            input_data = {}
            
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(features):
                # Check if feature is categorical (non-numeric)
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    # For categorical features, use selectbox with unique values
                    unique_values = df[feature].dropna().unique().tolist()
                    default_value = unique_values[0] if unique_values else ""
                    
                    # Place input fields in alternating columns
                    if i % 2 == 0:
                        input_data[feature] = col1.selectbox(
                            f"{feature}",
                            options=unique_values,
                            index=0
                        )
                    else:
                        input_data[feature] = col2.selectbox(
                            f"{feature}",
                            options=unique_values,
                            index=0
                        )
                else:
                    # For numeric features, use slider as before
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].median())
                    
                    # Place input fields in alternating columns
                    if i % 2 == 0:
                        input_data[feature] = col1.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=(max_val - min_val) / 100
                        )
                    else:
                        input_data[feature] = col2.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=(max_val - min_val) / 100
                        )
            
            # Make prediction
            if st.button("Predict Eligibility"):
                with st.spinner("Making prediction..."):
                    # Create input dataframe
                    input_df = pd.DataFrame([input_data])
                    
                    # Make prediction
                    prediction, probability = predict_eligibility(model, input_df)
                    
                    if prediction is not None:
                        # Determine eligibility status
                        eligible = prediction[0] == 1
                        
                        # Handle probability correctly depending on format
                        if isinstance(probability, np.ndarray) and probability.ndim == 2:
                            # If it's a 2D array of probabilities for each class
                            prob_pct = probability[0][1] * 100 if eligible else (1 - probability[0][1]) * 100
                        elif isinstance(probability, np.ndarray) and probability.ndim == 1:
                            # If it's a 1D array of probabilities for the positive class
                            prob_pct = probability[0] * 100 if eligible else (1 - probability[0]) * 100
                        else:
                            # Default case if probability format is unexpected
                            prob_pct = 100 if eligible else 0
                        
                        # Display prediction result with a gauge
                        st.markdown("### Eligibility Prediction Result")
                        
                        # Create a gauge chart for the probability
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob_pct,
                            title={'text': "Eligibility Probability"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "green" if eligible else "red"},
                                'steps': [
                                    {'range': [0, 50], 'color': 'lightcoral'},
                                    {'range': [50, 100], 'color': 'lightgreen'}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=50, b=20),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display eligibility verdict
                        if eligible:
                            st.success(f"✅ **ELIGIBLE** for donation with {prob_pct:.1f}% confidence")
                        else:
                            st.error(f"❌ **INELIGIBLE** for donation with {prob_pct:.1f}% confidence")
                        
                        # Provide additional recommendations for ineligible donors
                        if not eligible:
                            st.info("Recommendations for improving eligibility:")
                            st.markdown("""
                            - Ensure adequate iron levels and overall health
                            - Wait an appropriate time since last donation (typically 56 days)
                            - Maintain a healthy weight
                            - Ensure you meet minimum age requirements
                            - Consult with healthcare provider about any medical conditions
                            """)
                    else:
                        st.error("Unable to make prediction. Please check your input data or try retraining the model.")
            
            # Add disclaimer
            st.markdown("""
            **Disclaimer:** This prediction is based on statistical modeling and should be used as a guide only. 
            Final eligibility determination should always be made by qualified medical professionals in accordance with 
            official blood donation guidelines.
            """)

# Run the page function if this script is run directly
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv("../../blood_donation_data.csv")
    show_eligibility_prediction(df)
