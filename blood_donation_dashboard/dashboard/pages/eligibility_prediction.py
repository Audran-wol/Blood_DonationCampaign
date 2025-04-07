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
        feature_cols = [col for col in df.columns if col != 'Eligibility' and df[col].dtype != 'object']
        
        if len(feature_cols) > 0:
            # Allow manual feature selection
            with st.expander("Select Features for Model Training"):
                selected_features = st.multiselect(
                    "Select Features",
                    options=feature_cols,
                    default=feature_cols[:min(5, len(feature_cols))]
                )
            
            if len(selected_features) < 2:
                st.warning("Please select at least 2 features for model training.")
            else:
                # Train model on button click
                if st.button("Train and Evaluate Model"):
                    with st.spinner("Training model..."):
                        # Prepare data for modeling
                        X = df[selected_features]
                        y = df['Eligibility'].map({'Eligible': 1, 'Ineligible': 0})
                        
                        # Train model with selected configuration
                        model_name = selected_model.lower().replace(" ", "_")
                        
                        # Create a temporary DataFrame for training
                        model_df = X.copy()
                        model_df['target'] = y
                        
                        # Train the model
                        try:
                            model = train_model(model_df, model_type=model_name)
                            
                            # Evaluate model
                            evaluation_results = evaluate_model(model, X, y)
                            
                            # Display performance metrics
                            st.markdown("### Model Performance Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("Accuracy", f"{evaluation_results['accuracy']*100:.1f}%")
                            col2.metric("Precision", f"{evaluation_results['precision']*100:.1f}%")
                            col3.metric("Recall", f"{evaluation_results['recall']*100:.1f}%")
                            col4.metric("F1 Score", f"{evaluation_results['f1']*100:.1f}%")
                            
                            # Display ROC curve
                            if 'auc' in evaluation_results:
                                st.markdown("### ROC Curve")
                                roc_fig = visualize_model_performance(model)
                                st.plotly_chart(roc_fig, use_container_width=True)
                            
                            # Feature importance
                            if model and 'feature_importances' in model:
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
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                            st.info("Try selecting different features or a different model type.")
        else:
            st.error("No numeric features available for model training.")
    
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
                # Determine min, max, and default values based on the feature
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
                    
                    # Determine eligibility status
                    eligible = prediction[0] == 1
                    prob_pct = probability[0][1] * 100 if eligible else (1 - probability[0][1]) * 100
                    
                    # Display prediction result with a gauge
                    st.markdown("### Eligibility Prediction Result")
                    
                    # Create a gauge chart for the probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_pct,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkgreen" if eligible else "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "gray"},
                                {'range': [70, 100], 'color': "lightgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    
                    # Display eligibility result prominently
                    if eligible:
                        st.success(f"### ELIGIBLE ")
                        st.markdown(f"This donor is predicted to be **eligible** with {prob_pct:.1f}% confidence.")
                    else:
                        st.error(f"### INELIGIBLE ")
                        st.markdown(f"This donor is predicted to be **ineligible** with {prob_pct:.1f}% confidence.")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation of results
                    st.markdown("### Explanation")
                    
                    # If model provides feature importance, show contribution of each feature
                    if 'feature_importances' in model:
                        feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Importance': [model['feature_importances'][feat] for feat in features],
                            'Value': [input_data[feat] for feat in features]
                        })
                        
                        # Sort by importance
                        feature_importance = feature_importance.sort_values('Importance', ascending=False)
                        
                        # Show top contributing factors
                        st.markdown("#### Top Contributing Factors")
                        
                        for i, row in feature_importance.head(3).iterrows():
                            feature = row['Feature']
                            importance = row['Importance']
                            value = row['Value']
                            
                            # Determine if value is high or low compared to distribution
                            mean_val = df[feature].mean()
                            std_val = df[feature].std()
                            
                            if value > mean_val + std_val:
                                position = "high"
                            elif value < mean_val - std_val:
                                position = "low"
                            else:
                                position = "average"
                            
                            st.markdown(f"""
                            - **{feature}**: Value of {value:.2f} is **{position}** 
                              (importance: {importance:.3f})
                            """)
                        
                        # Provide recommendations if ineligible
                        if not eligible:
                            st.markdown("#### Recommendations to Improve Eligibility")
                            
                            for i, row in feature_importance.head(2).iterrows():
                                feature = row['Feature']
                                value = row['Value']
                                
                                # Simple recommendation based on whether higher or lower is better
                                # In a real application, this would be based on domain knowledge
                                corr = df[[feature, 'Eligibility']].corr().iloc[0, 1]
                                
                                if corr > 0:  # Positive correlation
                                    if value < df[feature].median():
                                        st.markdown(f"- Consider increasing **{feature}** (current value: {value:.2f})")
                                else:  # Negative correlation
                                    if value > df[feature].median():
                                        st.markdown(f"- Consider decreasing **{feature}** (current value: {value:.2f})")
            
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
