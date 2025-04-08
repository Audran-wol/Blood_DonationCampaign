"""
Blood Donation Dashboard - Main Application

This is the main entry point for the Blood Donation Dashboard application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path to import from scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from scripts.data_processing import load_data, clean_data, generate_features, get_geographical_data
from scripts.visualization import (create_map_visualization, create_donor_demographics, 
                                  create_health_conditions_vis, create_campaign_effectiveness_vis,
                                  create_donor_retention_vis, create_prediction_vis)
from scripts.clustering import perform_clustering, analyze_clusters
from scripts.prediction_model import load_model, predict_eligibility, visualize_model_performance

# Set page config
st.set_page_config(
    page_title="Blood Donation Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E53935;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0px 0px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .chart-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0px 0px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        background-color: #f0f0f0;
        border-radius: 5px 5px 0 0;
        color: #424242;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0;
        color: #E53935;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E53935 !important;
        color: white !important;
    }
    /* Fix white text on white background in visualization sections */
    .st-emotion-cache-1v0mbdj.e115fcil1, .st-emotion-cache-1t63c1f.e1f1d6gn1 {
        color: #424242 !important;
    }
    /* Make sure buttons have proper contrast */
    .stButton>button {
        background-color: #E53935;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
    }
    .stButton>button:hover {
        background-color: #C62828;
    }
    /* Improve metric display */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #E53935 !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #424242 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #424242 !important;
    }
    /* Fix background colors in KPI sections */
    .element-container:has([data-testid="stMetricValue"]) {
        background-color: #f8f8f8;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    /* Ensure proper contrast for links */
    a {
        color: #E53935;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    /* Better tooltip styling */
    div[data-baseweb="tooltip"] {
        background-color: #424242 !important;
        color: white !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading to improve performance
@st.cache_data
def get_processed_data():
    """Load and process the data."""
    # Load raw data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'blood_donation_data.csv')
    df = load_data(data_path)
    
    if df is not None:
        # Clean and preprocess
        df_clean = clean_data(df)
        # Generate additional features
        df_features = generate_features(df_clean)
        return df, df_clean, df_features
    else:
        st.error("Failed to load data. Please check the data file path.")
        return None, None, None

# Sidebar
st.sidebar.markdown('<h1 style="color:#E53935;">🩸 Blood Donation Dashboard</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Donor Distribution", "Health Conditions & Eligibility", 
     "Donor Profiling", "Campaign Effectiveness", "Donor Retention", "Eligibility Prediction"]
)

# Load data
with st.spinner("Loading data..."):
    raw_df, clean_df, feature_df = get_processed_data()

if raw_df is None or clean_df is None or feature_df is None:
    st.error("Error: Could not load or process data.")
    st.stop()

# Show data loading success
st.sidebar.success(f"Data loaded successfully: {len(raw_df)} records")

# Display data timestamp range
if 'Date de remplissage de la fiche' in feature_df.columns:
    date_col = 'Date de remplissage de la fiche'
    min_date = feature_df[date_col].min()
    max_date = feature_df[date_col].max()
    if not pd.isna(min_date) and not pd.isna(max_date):
        st.sidebar.info(f"Data period: {min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}")

# Add filters in the sidebar
st.sidebar.markdown("## Filters")

# Gender filter
genders = ['All']
if 'Gender' in feature_df.columns:
    genders.extend(feature_df['Gender'].dropna().unique())
selected_gender = st.sidebar.selectbox("Gender", genders)

# Age group filter
age_groups = ['All']
if 'Age_Group' in feature_df.columns:
    age_groups.extend(sorted(feature_df['Age_Group'].dropna().unique()))
selected_age_group = st.sidebar.selectbox("Age Group", age_groups)

# Previous donation filter
donation_options = ['All']
if 'Previous_Donation' in feature_df.columns:
    donation_options.extend(feature_df['Previous_Donation'].dropna().unique())
selected_donation = st.sidebar.selectbox("Previous Donation", donation_options)

# Eligibility filter
eligibility_options = ['All']
if 'Eligibility' in feature_df.columns:
    eligibility_options.extend(feature_df['Eligibility'].dropna().unique())
selected_eligibility = st.sidebar.selectbox("Eligibility", eligibility_options)

# Apply filters
filtered_df = feature_df.copy()

if selected_gender != 'All' and 'Gender' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

if selected_age_group != 'All' and 'Age_Group' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Age_Group'] == selected_age_group]

if selected_donation != 'All' and 'Previous_Donation' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Previous_Donation'] == selected_donation]

if selected_eligibility != 'All' and 'Eligibility' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Eligibility'] == selected_eligibility]

# Display record count after filtering
st.sidebar.markdown(f"**Filtered records:** {len(filtered_df)}")
st.sidebar.markdown("---")

# About section in sidebar
with st.sidebar.expander("About this Dashboard"):
    st.markdown("""
    This dashboard provides comprehensive analytics and visualization tools for blood donation campaign data.
    
    It allows you to:
    - Analyze donor demographics and geographic distribution
    - Examine health conditions and eligibility factors
    - Profile ideal donors through clustering
    - Evaluate campaign effectiveness
    - Analyze donor retention patterns
    - Predict donor eligibility
    
    For more information, please refer to the documentation.
    """)

# Main content based on selected page
if page == "Overview":
    # Header
    st.markdown('<div class="main-header">Blood Donation Campaign Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Welcome to the comprehensive blood donation campaign analytics dashboard.")
    st.markdown("---")
    
    # Key metrics section
    st.markdown('<div class="sub-header">Key Metrics</div>', unsafe_allow_html=True)
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Total donors
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Donors", f"{len(raw_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Eligibility rate
    with col2:
        if 'Eligibility' in feature_df.columns:
            eligible_count = feature_df['Eligibility'].value_counts().get('Eligible', 0)
            eligible_percent = (eligible_count / len(feature_df)) * 100 if len(feature_df) > 0 else 0
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Eligibility Rate", f"{eligible_percent:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Eligibility Rate", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Gender distribution
    with col3:
        if 'Gender' in feature_df.columns:
            male_count = feature_df['Gender'].value_counts().get('Male', 0)
            male_percent = (male_count / len(feature_df)) * 100 if len(feature_df) > 0 else 0
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Male Donors", f"{male_percent:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Male Donors", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Previous donation
    with col4:
        if 'Previous_Donation' in feature_df.columns:
            repeat_count = feature_df['Previous_Donation'].value_counts().get('Yes', 0)
            repeat_percent = (repeat_count / len(feature_df)) * 100 if len(feature_df) > 0 else 0
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Repeat Donors", f"{repeat_percent:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Repeat Donors", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick visual summaries
    st.markdown('<div class="sub-header">Quick Visual Summaries</div>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Eligibility Factors", "Geographic Distribution", "Time Trends"])
    
    with tab1:
        # Create demographic visualizations
        demo_figs = create_donor_demographics(filtered_df)
        
        # Display demographic visualizations
        cols = st.columns(2)
        
        if 'age_distribution' in demo_figs:
            cols[0].plotly_chart(demo_figs['age_distribution'], use_container_width=True)
        
        if 'gender_distribution' in demo_figs:
            cols[1].plotly_chart(demo_figs['gender_distribution'], use_container_width=True)
        
        cols = st.columns(2)
        
        if 'education_distribution' in demo_figs:
            cols[0].plotly_chart(demo_figs['education_distribution'], use_container_width=True)
        
        if 'profession_distribution' in demo_figs:
            cols[1].plotly_chart(demo_figs['profession_distribution'], use_container_width=True)
    
    with tab2:
        # Create health condition visualizations
        health_figs = create_health_conditions_vis(filtered_df)
        
        # Display health condition visualizations
        if health_figs:
            cols = st.columns(2)
            
            if 'health_eligibility' in health_figs:
                cols[0].plotly_chart(health_figs['health_eligibility'], use_container_width=True)
            
            if 'risk_eligibility' in health_figs:
                cols[1].plotly_chart(health_figs['risk_eligibility'], use_container_width=True)
        else:
            st.info("No health condition data available for visualization.")
    
    with tab3:
        # Create map visualization
        geo_data = get_geographical_data(filtered_df)
        if geo_data is not None:
            map_fig = create_map_visualization(geo_data)
            if map_fig is not None:
                st.plotly_chart(map_fig, use_container_width=True)
            else:
                st.info("Could not create map visualization from the available data.")
        else:
            st.info("No geographical data available for visualization.")
    
    with tab4:
        # Create time trend visualizations
        trend_figs = create_campaign_effectiveness_vis(filtered_df)
        
        # Display time trend visualizations
        if trend_figs:
            cols = st.columns(2)
            
            if 'donation_trend' in trend_figs:
                cols[0].plotly_chart(trend_figs['donation_trend'], use_container_width=True)
            
            if 'donation_month' in trend_figs:
                cols[1].plotly_chart(trend_figs['donation_month'], use_container_width=True)
                
            # Display blood type and phenotype distributions
            st.subheader("Blood Type Analysis")
            bt_cols = st.columns(2)
            
            if 'blood_type_distribution' in trend_figs:
                bt_cols[0].plotly_chart(trend_figs['blood_type_distribution'], use_container_width=True)
            
            if 'abo_distribution' in trend_figs:
                bt_cols[1].plotly_chart(trend_figs['abo_distribution'], use_container_width=True)
                
            if 'phenotype_distribution' in trend_figs:
                st.plotly_chart(trend_figs['phenotype_distribution'], use_container_width=True)
        else:
            st.info("No time trend data available for visualization.")

elif page == "Donor Distribution":
    # Import the donor distribution page
    from pages.donor_distribution import show_donor_distribution
    
    # Show the donor distribution page with the filtered data
    show_donor_distribution(filtered_df)

elif page == "Health Conditions & Eligibility":
    # Import the health conditions page
    from pages.health_conditions import show_health_conditions
    
    # Show the health conditions page with the filtered data
    show_health_conditions(filtered_df)

elif page == "Donor Profiling":
    # Import the donor profiling page
    from pages.donor_profiling import show_donor_profiling
    
    # Show the donor profiling page with the filtered data
    show_donor_profiling(filtered_df)

elif page == "Campaign Effectiveness":
    # Import the campaign effectiveness page
    from pages.campaign_effectiveness import show_campaign_effectiveness
    
    # Show the campaign effectiveness page with the filtered data
    show_campaign_effectiveness(filtered_df)

elif page == "Donor Retention":
    # Import the donor retention page
    from pages.donor_retention import show_donor_retention
    
    # Show the donor retention page with the filtered data
    show_donor_retention(filtered_df)

elif page == "Eligibility Prediction":
    # Import the eligibility prediction page
    from pages.eligibility_prediction import show_eligibility_prediction
    
    # Show the eligibility prediction page with the filtered data
    show_eligibility_prediction(filtered_df)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #616161; padding: 10px;">
        Blood Donation Dashboard &copy; 2025 | Data last updated: {}
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")),
    unsafe_allow_html=True
)
