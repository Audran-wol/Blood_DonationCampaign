"""
Donor Distribution Page

This module displays the geographical and demographic distribution of blood donors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Add parent directory to path to import from scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from scripts.data_processing import get_geographical_data
from scripts.visualization import create_map_visualization, create_donor_demographics

def show_donor_distribution(df):
    """
    Display the donor distribution page.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Filtered dataframe with donor information
    """
    # Header
    st.markdown('<div class="main-header">Donor Distribution Analysis</div>', unsafe_allow_html=True)
    st.markdown("Analyze the geographical and demographic distribution of blood donors.")
    st.markdown("---")
    
    # Create tabs for different visualization categories
    tab1, tab2 = st.tabs(["Geographic Distribution", "Demographic Distribution"])
    
    with tab1:
        st.markdown('<div class="sub-header">Geographic Distribution of Donors</div>', unsafe_allow_html=True)
        
        # Create columns for filters and map
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### Filters")
            
            # Add geographic filters if applicable
            location_cols = ['District', 'Neighborhood']
            geo_filters = {}
            
            for col in location_cols:
                if col in df.columns:
                    unique_vals = sorted(df[col].dropna().unique())
                    if len(unique_vals) > 0:
                        selected = st.multiselect(f"Select {col}", unique_vals)
                        if selected:
                            geo_filters[col] = selected
            
            # Allow filtering by eligibility status
            if 'Eligibility' in df.columns:
                eligibility_options = sorted(df['Eligibility'].dropna().unique())
                selected_elig = st.multiselect("Eligibility Status", eligibility_options)
                if selected_elig:
                    geo_filters['Eligibility'] = selected_elig
            
            # Apply geographic filters
            geo_filtered_df = df.copy()
            for col, values in geo_filters.items():
                if values:
                    geo_filtered_df = geo_filtered_df[geo_filtered_df[col].isin(values)]
            
            # Display donor count after filtering
            st.metric("Filtered Donors", f"{len(geo_filtered_df):,}")
            
            # Add a download button for the filtered data
            csv_data = geo_filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Data",
                data=csv_data,
                file_name="filtered_donor_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create map visualization
            with st.spinner("Creating map visualization..."):
                geo_data = get_geographical_data(geo_filtered_df)
                
                if geo_data is not None and not geo_data.empty:
                    map_fig = create_map_visualization(geo_data)
                    
                    if map_fig is not None:
                        st.plotly_chart(map_fig, use_container_width=True)
                    else:
                        st.error("Failed to create map visualization.")
                else:
                    st.info("No geographical data available for the selected filters.")
        
        # Add statistics about geographic distribution
        st.markdown('<div class="sub-header">Geographic Distribution Statistics</div>', unsafe_allow_html=True)
        
        cols = st.columns(2)
        
        # Top districts by donor count
        if 'District' in df.columns:
            district_counts = geo_filtered_df['District'].value_counts().reset_index()
            district_counts.columns = ['District', 'Donor Count']
            
            # Take top 10 districts
            top_districts = district_counts.head(10)
            
            fig_districts = px.bar(
                top_districts,
                x='District',
                y='Donor Count',
                title='Top 10 Districts by Donor Count',
                color='Donor Count',
                color_continuous_scale=px.colors.sequential.Reds
            )
            
            cols[0].plotly_chart(fig_districts, use_container_width=True)
        
        # Top neighborhoods by donor count
        if 'Neighborhood' in df.columns:
            neighborhood_counts = geo_filtered_df['Neighborhood'].value_counts().reset_index()
            neighborhood_counts.columns = ['Neighborhood', 'Donor Count']
            
            # Take top 10 neighborhoods
            top_neighborhoods = neighborhood_counts.head(10)
            
            fig_neighborhoods = px.bar(
                top_neighborhoods,
                x='Neighborhood',
                y='Donor Count',
                title='Top 10 Neighborhoods by Donor Count',
                color='Donor Count',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            cols[1].plotly_chart(fig_neighborhoods, use_container_width=True)
        
        # Eligibility by location (if applicable)
        if 'District' in df.columns and 'Eligibility' in df.columns:
            # Cross-tabulate district and eligibility
            district_elig = pd.crosstab(
                df['District'], 
                df['Eligibility'],
                normalize='index'
            ) * 100
            
            if 'Eligible' in district_elig.columns:
                district_elig = district_elig.reset_index()
                district_elig = district_elig.sort_values('Eligible', ascending=False).head(10)
                
                fig_district_elig = px.bar(
                    district_elig,
                    x='District',
                    y='Eligible',
                    title='Top 10 Districts by Eligibility Rate',
                    labels={
                        'District': 'District',
                        'Eligible': 'Eligible Percentage (%)'
                    },
                    color='Eligible',
                    color_continuous_scale=px.colors.sequential.Greens
                )
                
                st.plotly_chart(fig_district_elig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Demographic Distribution of Donors</div>', unsafe_allow_html=True)
        
        # Create demographic visualizations
        demo_figs = create_donor_demographics(df)
        
        # Display demographic visualizations in a 2x2 grid
        if demo_figs:
            grid1_col1, grid1_col2 = st.columns(2)
            
            if 'age_distribution' in demo_figs:
                grid1_col1.plotly_chart(demo_figs['age_distribution'], use_container_width=True)
            
            if 'gender_distribution' in demo_figs:
                grid1_col2.plotly_chart(demo_figs['gender_distribution'], use_container_width=True)
            
            grid2_col1, grid2_col2 = st.columns(2)
            
            if 'education_distribution' in demo_figs:
                grid2_col1.plotly_chart(demo_figs['education_distribution'], use_container_width=True)
            
            if 'profession_distribution' in demo_figs:
                grid2_col2.plotly_chart(demo_figs['profession_distribution'], use_container_width=True)
            
            # Additional demographic analysis if available
            if 'age_gender_distribution' in demo_figs:
                st.plotly_chart(demo_figs['age_gender_distribution'], use_container_width=True)
        else:
            st.info("No demographic data available for visualization.")
        
        # Add demographic analysis by location
        st.markdown('<div class="sub-header">Demographics by Location</div>', unsafe_allow_html=True)
        
        # Select a location type
        location_type = st.selectbox(
            "Select Location Type",
            ["District", "Neighborhood"],
            index=0
        )
        
        if location_type in df.columns:
            # Select specific locations
            locations = sorted(df[location_type].dropna().unique())
            selected_locations = st.multiselect(
                f"Select {location_type}s to Compare",
                locations,
                default=locations[:3] if len(locations) >= 3 else locations
            )
            
            if selected_locations:
                # Filter data for selected locations
                location_df = df[df[location_type].isin(selected_locations)]
                
                # Create comparison visualizations
                cols = st.columns(2)
                
                # Age distribution by location
                if 'Age' in location_df.columns:
                    fig_age_loc = px.box(
                        location_df,
                        x=location_type,
                        y='Age',
                        color=location_type,
                        title=f'Age Distribution by {location_type}',
                        labels={
                            location_type: location_type,
                            'Age': 'Age (years)'
                        }
                    )
                    
                    cols[0].plotly_chart(fig_age_loc, use_container_width=True)
                
                # Gender distribution by location
                if 'Gender' in location_df.columns:
                    gender_loc = pd.crosstab(
                        location_df[location_type],
                        location_df['Gender'],
                        normalize='index'
                    ) * 100
                    
                    gender_loc = gender_loc.reset_index()
                    
                    fig_gender_loc = px.bar(
                        gender_loc,
                        x=location_type,
                        y=gender_loc.columns[1:].tolist(),
                        title=f'Gender Distribution by {location_type}',
                        labels={
                            'value': 'Percentage (%)',
                            'variable': 'Gender'
                        },
                        barmode='group'
                    )
                    
                    cols[1].plotly_chart(fig_gender_loc, use_container_width=True)
            else:
                st.info(f"Please select at least one {location_type} to compare demographics.")
