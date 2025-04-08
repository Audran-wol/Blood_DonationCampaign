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
from scripts.prediction_model import train_model, evaluate_model, predict_eligibility, train_eligibility_model
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
            
            # Define automatic disqualifiers to exclude from the input form
            automatic_disqualifiers = [
                'Porteur(HIV,hbs,hcv)',  # HIV, Hepatitis B/C carriers
                'Drepanocytaire',        # Sickle cell disease
            ]
            
            # Create a notice about automatic disqualifications
            st.info("""
            ### Medical Safety Information
            Certain medical conditions automatically disqualify a person from donating blood 
            for the safety of both donors and recipients. These include:
            - HIV, Hepatitis B, or Hepatitis C positive status
            - Sickle Cell Disease
            
            These conditions are not included in the prediction form as they result in automatic disqualification.
            """)
            
            # Filter out the automatic disqualifiers from features
            filtered_features = [f for f in features if f not in automatic_disqualifiers]
            
            for i, feature in enumerate(filtered_features):
                # Identify health condition features that should be boolean/categorical
                health_conditions = ['Antécédent_de_transfusion', 'Opéré',  
                                    'Diabétique', 'Hypertendus', 'Asthmatiques',
                                    'Cardiaque', 'Tatoué', 'Scarifié']
                
                # Check if feature is categorical (non-numeric) or should be treated as categorical
                if feature in health_conditions or not pd.api.types.is_numeric_dtype(df[feature]):
                    # For categorical features, use selectbox with appropriate values
                    if feature in health_conditions:
                        # Health conditions should be Yes/No choices
                        options = ['Non', 'Oui'] if 'Oui' in df[feature].unique() else ['No', 'Yes']
                        default_idx = 0  # Default to 'No'/'Non'
                    else:
                        # For other categorical features, use unique values from dataset
                        unique_values = df[feature].dropna().unique().tolist()
                        options = unique_values
                        default_idx = 0 if unique_values else 0
                    
                    # Place input fields in alternating columns
                    if i % 2 == 0:
                        input_data[feature] = col1.selectbox(
                            f"{feature}",
                            options=options,
                            index=default_idx,
                            help="Select appropriate value"
                        )
                    else:
                        input_data[feature] = col2.selectbox(
                            f"{feature}",
                            options=options,
                            index=default_idx,
                            help="Select appropriate value"
                        )
                # Special handling for Health_Risk_Score
                elif feature == 'Health_Risk_Score':
                    min_val = 0.0  # Health risk score should start at 0
                    max_val = float(df[feature].max())
                    default_val = 0.0  # Default to 0 (healthy)
                    
                    if i % 2 == 0:
                        input_data[feature] = col1.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=1.0,  # Use whole numbers for risk score
                            help="Health risk score (higher values indicate greater health risks)"
                        )
                    else:
                        input_data[feature] = col2.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=1.0,  # Use whole numbers for risk score
                            help="Health risk score (higher values indicate greater health risks)"
                        )
                else:
                    # For other numeric features, use slider
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].median())
                    
                    # For age, use whole numbers
                    step = 1.0 if feature == 'Age' else (max_val - min_val) / 100
                    
                    # Place input fields in alternating columns
                    if i % 2 == 0:
                        input_data[feature] = col1.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=step,
                            help=f"Enter value for {feature}"
                        )
                    else:
                        input_data[feature] = col2.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=step,
                            help=f"Enter value for {feature}"
                        )
            
            # Add a button to make predictions
            if st.button("Predict Eligibility"):
                # Train model with selected features
                model_result = train_eligibility_model(df, model_type=selected_model, selected_features=features)
                
                if model_result:
                    # Get prediction from model
                    def predict_with_input_data(model_result, input_data):
                        """Get prediction for the input data"""
                        if model_result is None:
                            st.error("Model not loaded. Please train the model first.")
                            return None, None, None
                        
                        # Make prediction
                        try:
                            prediction, probability, explanation = predict_eligibility(model_result, input_data)
                            return prediction, probability, explanation
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")
                            return None, None, None
                    
                    # Display prediction result
                    def display_prediction_result(prediction, probability, explanation, feature_importances=None):
                        """Display the prediction result with explanation"""
                        if prediction is None:
                            st.error("Could not make a prediction. Please check the input data.")
                            return
                        
                        # Get formatted probability
                        prob_formatted = f"{probability:.1%}" if probability is not None else "Unknown"
                        
                        # Create visually appealing prediction display
                        col1, col2 = st.columns([1, 3])
                        
                        # Check if this is an automatic disqualification
                        is_auto_disqualification = "Automatic disqualification" in explanation if explanation else False
                        
                        with col1:
                            # Display a circular indicator with percentage
                            if prediction == 1 and not is_auto_disqualification:
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=probability*100,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Eligibility"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "green"},
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 50
                                        }
                                    }
                                ))
                            else:
                                # For ineligible or auto-disqualified, always show 0%
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=0,  # Always 0 for ineligible
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Eligibility"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "red"},
                                        'threshold': {
                                            'line': {'color': "green", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 50
                                        }
                                    }
                                ))
                            
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Display prediction text with explanation
                            if prediction == 1 and not is_auto_disqualification:
                                st.markdown(f"### Prediction: **ELIGIBLE**")
                                st.markdown(f"Confidence: **{prob_formatted}**")
                            else:
                                st.markdown(f"### Prediction: **NOT ELIGIBLE**")
                                if is_auto_disqualification:
                                    st.markdown("### ⚠️ MEDICAL DISQUALIFICATION ⚠️")
                                    st.markdown(f"**{explanation}**")
                                else:
                                    st.markdown(f"Confidence: **{prob_formatted}**")
                            
                            # Display explanation in expander
                            with st.expander("View Explanation"):
                                if explanation:
                                    st.write(explanation)
                                else:
                                    st.write("No explanation available.")
                            
                            # Display feature importances if available (only for model-based predictions)
                            if not is_auto_disqualification and feature_importances and isinstance(feature_importances, dict) and len(feature_importances) > 0:
                                with st.expander("View Feature Importance Chart"):
                                    importances = pd.DataFrame({
                                        'Feature': list(feature_importances.keys()),
                                        'Importance': list(feature_importances.values())
                                    })
                                    
                                    # Sort by importance
                                    importances = importances.sort_values('Importance', ascending=False).head(10)
                                    
                                    # Create bar chart
                                    fig = px.bar(
                                        importances, 
                                        x='Importance', 
                                        y='Feature', 
                                        orientation='h',
                                        title='Top 10 Feature Importances'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Process the prediction and display results
                    def process_prediction(model_result, input_data):
                        """Process the prediction and display results"""
                        # Set automatic disqualifiers to 'No' by default if not in input_data
                        # This means missing fields won't trigger disqualification
                        disqualifiers = {
                            'Porteur(HIV,hbs,hcv)': 'Non',  # Default to 'No'
                            'Drepanocytaire': 'Non'         # Default to 'No'
                        }
                        
                        # Update input_data with default values for any missing disqualifiers
                        for disqualifier, default_value in disqualifiers.items():
                            if disqualifier not in input_data:
                                input_data[disqualifier] = default_value
                        
                        # Get prediction
                        prediction, probability, explanation = predict_with_input_data(model_result, input_data)
                        
                        # Display prediction result
                        if prediction is not None:
                            # Calculate feature importances if possible
                            feature_importances = None
                            if model_result and 'feature_importances' in model_result:
                                feature_importances = model_result['feature_importances']
                            
                            # Display the prediction result
                            display_prediction_result(prediction, probability, explanation, feature_importances)
                            
                            # Provide additional recommendations for ineligible donors
                            is_auto_disqualification = "Automatic disqualification" in explanation if explanation else False
                            if prediction == 0 and not is_auto_disqualification:
                                st.info("Recommendations for improving eligibility:")
                                st.markdown("""
                                - Ensure adequate iron levels and overall health
                                - Wait an appropriate time since last donation (typically 56 days)
                                - Maintain a healthy weight
                                - Ensure you meet minimum age requirements
                                - Consult with healthcare provider about any medical conditions
                                """)
                    
                    # Make prediction
                    process_prediction(model_result, input_data)
                    
                    # Add disclaimer
                    st.markdown("""
                    **Disclaimer:** This prediction is based on statistical modeling and should be used as a guide only. 
                    Final eligibility determination should always be made by qualified medical professionals in accordance with 
                    official blood donation guidelines.
                    """)
                else:
                    st.error("Model training failed. Please check the console for details.")
                    
# Run the page function if this script is run directly
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv("../../blood_donation_data.csv")
    show_eligibility_prediction(df)
