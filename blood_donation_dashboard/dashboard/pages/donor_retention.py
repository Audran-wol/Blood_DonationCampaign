"""
Donor Retention Page

This module analyzes the patterns and factors affecting donor retention.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path to import from scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from scripts.visualization import create_donor_retention_vis

def show_donor_retention(df):
    """
    Display the donor retention analysis page.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Filtered dataframe with donor information
    """
    # Header
    st.markdown('<div class="main-header">Donor Retention Analysis</div>', unsafe_allow_html=True)
    st.markdown("Analyze patterns and factors affecting the return rate of blood donors.")
    st.markdown("---")
    
    # Check if we have previous donation information
    has_previous_donation = 'Previous_Donation' in df.columns
    
    if not has_previous_donation:
        st.warning("Previous donation information is not available in the dataset for retention analysis.")
    
    # Create tabs for different aspects of donor retention
    tab1, tab2, tab3 = st.tabs(["Retention Overview", "Demographic Factors", "Engagement Analysis"])
    
    with tab1:
        st.markdown('<div class="sub-header">Donor Retention Overview</div>', unsafe_allow_html=True)
        
        if has_previous_donation:
            # Calculate retention metrics
            total_donors = len(df)
            returning_donors = df['Previous_Donation'].value_counts().get('Yes', 0)
            new_donors = df['Previous_Donation'].value_counts().get('No', 0)
            
            retention_rate = (returning_donors / total_donors) * 100 if total_donors > 0 else 0
            
            # Display retention metrics
            col1, col2, col3 = st.columns(3)
            
            col1.metric(
                "Overall Retention Rate",
                f"{retention_rate:.1f}%",
                help="Percentage of donors who have donated before"
            )
            
            col2.metric(
                "Returning Donors",
                f"{returning_donors:,}",
                f"{(returning_donors / total_donors) * 100:.1f}%" if total_donors > 0 else None,
                help="Number of donors who have donated before"
            )
            
            col3.metric(
                "New Donors",
                f"{new_donors:,}",
                f"{(new_donors / total_donors) * 100:.1f}%" if total_donors > 0 else None,
                help="Number of first-time donors"
            )
            
            # Create donor retention visualizations
            retention_figs = create_donor_retention_vis(df)
            
            if retention_figs:
                # Display retention visualizations
                for fig_name, fig in retention_figs.items():
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display retention over time if date information is available
            date_col = None
            date_cols = ['Date de remplissage de la fiche', 'Campaign_Date', 'Donation_Date']
            
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col and pd.api.types.is_datetime64_dtype(df[date_col]):
                st.markdown("### Retention Rate Over Time")
                
                try:
                    # Calculate retention by time period
                    df['YearMonth'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m')
                    
                    # Calculate retention by time period
                    retention_by_time = df.groupby('YearMonth')['Previous_Donation'].apply(
                        lambda x: (x == 'Yes').mean() * 100
                    ).reset_index()
                    
                    # Convert YearMonth to datetime for proper sorting and plotting
                    retention_by_time['YearMonth_dt'] = pd.to_datetime(retention_by_time['YearMonth'] + '-01')
                    
                    # Sort by year and month for proper chronological display
                    retention_by_time = retention_by_time.sort_values('YearMonth_dt')
                    
                    # Create line chart
                    fig = px.line(
                        retention_by_time,
                        x='YearMonth',  # Keep using the string version for display
                        y='Previous_Donation',
                        title='Donor Retention Rate by Month',
                        labels={
                            'YearMonth': 'Month',
                            'Previous_Donation': 'Retention Rate (%)'
                        },
                        markers=True
                    )
                    
                    # Add trend line
                    fig.add_traces(
                        px.scatter(
                            retention_by_time,
                            x='YearMonth',
                            y='Previous_Donation',
                            trendline='lowess',
                            trendline_color_override='red'
                        ).data[1]
                    )
                    
                    # Add target threshold line at 50%
                    fig.add_shape(
                        type='line',
                        x0=0,
                        y0=50,
                        x1=len(retention_by_time) - 1,
                        y1=50,
                        line=dict(color='green', width=2, dash='dash')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating retention time series: {e}")
            
            # Retention by eligibility status
            if 'Eligibility' in df.columns:
                st.markdown("### Retention by Eligibility Status")
                
                # Calculate retention by eligibility
                retention_by_eligibility = pd.crosstab(
                    df['Eligibility'],
                    df['Previous_Donation'],
                    normalize='index'
                ) * 100
                
                retention_by_eligibility = retention_by_eligibility.reset_index()
                
                # Create bar chart
                if 'Yes' in retention_by_eligibility.columns:
                    fig = px.bar(
                        retention_by_eligibility,
                        x='Eligibility',
                        y='Yes',
                        title='Retention Rate by Eligibility Status',
                        labels={
                            'Eligibility': 'Eligibility Status',
                            'Yes': 'Retention Rate (%)'
                        },
                        color='Yes',
                        text='Yes',
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add insight
                    if len(retention_by_eligibility) == 2:
                        eligible_rate = retention_by_eligibility.loc[
                            retention_by_eligibility['Eligibility'] == 'Eligible', 'Yes'
                        ].iloc[0]
                        
                        ineligible_rate = retention_by_eligibility.loc[
                            retention_by_eligibility['Eligibility'] == 'Ineligible', 'Yes'
                        ].iloc[0]
                        
                        diff = eligible_rate - ineligible_rate
                        
                        if diff > 0:
                            st.markdown(f"""
                            **Insight:** Eligible donors have a {diff:.1f} percentage points higher retention rate 
                            compared to ineligible donors. This suggests that positive donation experiences may 
                            encourage donors to return.
                            """)
                        else:
                            st.markdown(f"""
                            **Insight:** Interestingly, ineligible donors have a {-diff:.1f} percentage points higher 
                            retention rate compared to eligible donors. This could suggest persistent motivation 
                            despite previous ineligibility.
                            """)
            
            # Add retention benchmark
            st.markdown("### Retention Benchmarks")
            
            # Create a gauge chart for retention rate compared to benchmark
            benchmark_retention = 45.0  # Industry benchmark (example value)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=retention_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Retention Rate vs. Benchmark"},
                delta={'reference': benchmark_retention},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 50], 'color': "orange"},
                        {'range': [50, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': benchmark_retention
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Industry Benchmarks:**
            - **Low Retention**: < 30%
            - **Average Retention**: 30-50%
            - **High Retention**: > 50%
            
            *Note: Actual benchmarks may vary by region and campaign type.*
            """)
        else:
            st.info("Previous donation information is required for retention analysis.")
    
    with tab2:
        st.markdown('<div class="sub-header">Demographic Factors in Retention</div>', unsafe_allow_html=True)
        
        if has_previous_donation:
            # Analyze retention by demographic factors
            
            # Age group and retention
            if 'Age_Group' in df.columns:
                st.markdown("### Retention by Age Group")
                
                # Calculate retention by age group
                retention_by_age = pd.crosstab(
                    df['Age_Group'],
                    df['Previous_Donation'],
                    normalize='index'
                ) * 100
                
                retention_by_age = retention_by_age.reset_index()
                
                # Create bar chart
                if 'Yes' in retention_by_age.columns:
                    # Sort by age group if possible
                    try:
                        age_order = sorted(retention_by_age['Age_Group'].unique(), 
                                          key=lambda x: int(x.split('-')[0]) if '-' in x else 0)
                        retention_by_age['Age_Group'] = pd.Categorical(
                            retention_by_age['Age_Group'], 
                            categories=age_order, 
                            ordered=True
                        )
                        retention_by_age = retention_by_age.sort_values('Age_Group')
                    except:
                        pass
                    
                    fig = px.bar(
                        retention_by_age,
                        x='Age_Group',
                        y='Yes',
                        title='Retention Rate by Age Group',
                        labels={
                            'Age_Group': 'Age Group',
                            'Yes': 'Retention Rate (%)'
                        },
                        color='Yes',
                        text='Yes',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify best age group for retention
                    best_age_group = retention_by_age.loc[retention_by_age['Yes'].idxmax(), 'Age_Group']
                    best_retention = retention_by_age['Yes'].max()
                    
                    st.markdown(f"""
                    **Insight:** The {best_age_group} age group has the highest retention rate at {best_retention:.1f}%. 
                    This suggests focusing recruitment and retention efforts on this demographic could yield better results.
                    """)
            
            # Gender and retention
            if 'Gender' in df.columns:
                st.markdown("### Retention by Gender")
                
                # Calculate retention by gender
                retention_by_gender = pd.crosstab(
                    df['Gender'],
                    df['Previous_Donation'],
                    normalize='index'
                ) * 100
                
                retention_by_gender = retention_by_gender.reset_index()
                
                # Create bar chart
                if 'Yes' in retention_by_gender.columns:
                    fig = px.bar(
                        retention_by_gender,
                        x='Gender',
                        y='Yes',
                        title='Retention Rate by Gender',
                        labels={
                            'Gender': 'Gender',
                            'Yes': 'Retention Rate (%)'
                        },
                        color='Gender',
                        text='Yes'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display gap
                    if len(retention_by_gender) == 2:
                        gender_values = retention_by_gender['Gender'].tolist()
                        if 'Male' in gender_values and 'Female' in gender_values:
                            male_rate = retention_by_gender.loc[
                                retention_by_gender['Gender'] == 'Male', 'Yes'
                            ].iloc[0]
                            
                            female_rate = retention_by_gender.loc[
                                retention_by_gender['Gender'] == 'Female', 'Yes'
                            ].iloc[0]
                            
                            gap = abs(male_rate - female_rate)
                            
                            if male_rate > female_rate:
                                st.markdown(f"""
                                **Gender Gap:** Male donors have a {gap:.1f} percentage points higher retention rate 
                                compared to female donors.
                                """)
                            else:
                                st.markdown(f"""
                                **Gender Gap:** Female donors have a {gap:.1f} percentage points higher retention rate 
                                compared to male donors.
                                """)
            
            # Education and retention
            if 'Education' in df.columns:
                st.markdown("### Retention by Education Level")
                
                # Calculate retention by education
                retention_by_education = pd.crosstab(
                    df['Education'],
                    df['Previous_Donation'],
                    normalize='index'
                ) * 100
                
                retention_by_education = retention_by_education.reset_index()
                
                # Create bar chart
                if 'Yes' in retention_by_education.columns:
                    fig = px.bar(
                        retention_by_education,
                        x='Education',
                        y='Yes',
                        title='Retention Rate by Education Level',
                        labels={
                            'Education': 'Education Level',
                            'Yes': 'Retention Rate (%)'
                        },
                        color='Yes',
                        text='Yes',
                        color_continuous_scale='Oranges'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Location and retention
            location_cols = ['District', 'Neighborhood']
            location_col = None
            
            for col in location_cols:
                if col in df.columns:
                    location_col = col
                    break
            
            if location_col:
                st.markdown(f"### Retention by {location_col}")
                
                # Calculate retention by location
                retention_by_location = pd.crosstab(
                    df[location_col],
                    df['Previous_Donation'],
                    normalize='index'
                ) * 100
                
                retention_by_location = retention_by_location.reset_index()
                
                # Create bar chart with top 10 locations by donor count
                if 'Yes' in retention_by_location.columns:
                    # Get top 10 locations by donor count
                    top_locations = df[location_col].value_counts().head(10).index.tolist()
                    filtered_locations = retention_by_location[retention_by_location[location_col].isin(top_locations)]
                    
                    # Sort by retention rate
                    filtered_locations = filtered_locations.sort_values('Yes', ascending=False)
                    
                    fig = px.bar(
                        filtered_locations,
                        x=location_col,
                        y='Yes',
                        title=f'Retention Rate by Top 10 {location_col}s',
                        labels={
                            location_col: location_col,
                            'Yes': 'Retention Rate (%)'
                        },
                        color='Yes',
                        text='Yes',
                        color_continuous_scale='Greens'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify best location for retention
                    best_location = filtered_locations.iloc[0][location_col]
                    best_retention = filtered_locations.iloc[0]['Yes']
                    
                    st.markdown(f"""
                    **Insight:** {best_location} has the highest retention rate at {best_retention:.1f}% among the top 10 
                    locations with the most donors. This could indicate successful local engagement strategies that could be 
                    applied elsewhere.
                    """)
            
            # Multivariate analysis
            st.markdown("### Multivariate Analysis of Retention Factors")
            
            # Select factors for analysis
            demographic_cols = ['Gender', 'Age_Group', 'Education']
            available_cols = [col for col in demographic_cols if col in df.columns]
            
            if len(available_cols) >= 2:
                # Select two factors for comparison
                col1, col2 = available_cols[:2]
                
                # Calculate retention by two factors
                retention_by_factors = pd.crosstab(
                    [df[col1], df[col2]],
                    df['Previous_Donation'],
                    normalize='index'
                ) * 100
                
                retention_by_factors = retention_by_factors.reset_index()
                
                # Create grouped bar chart
                if 'Yes' in retention_by_factors.columns:
                    # Create a new column for labels
                    retention_by_factors['Combined'] = retention_by_factors[col1] + ' - ' + retention_by_factors[col2]
                    
                    fig = px.bar(
                        retention_by_factors,
                        x=col1,
                        y='Yes',
                        color=col2,
                        title=f'Retention Rate by {col1} and {col2}',
                        labels={
                            col1: col1,
                            col2: col2,
                            'Yes': 'Retention Rate (%)'
                        },
                        barmode='group',
                        text='Yes'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Highlight highest and lowest retention combinations
                    highest_combo = retention_by_factors.loc[retention_by_factors['Yes'].idxmax()]
                    lowest_combo = retention_by_factors.loc[retention_by_factors['Yes'].idxmin()]
                    
                    st.markdown(f"""
                    **Highest Retention:** {highest_combo[col1]} {highest_combo[col2]} donors have the highest retention rate at {highest_combo['Yes']:.1f}%.
                    
                    **Lowest Retention:** {lowest_combo[col1]} {lowest_combo[col2]} donors have the lowest retention rate at {lowest_combo['Yes']:.1f}%.
                    
                    This suggests targeting specific demographic combinations may be more effective for retention strategies.
                    """)
        else:
            st.info("Previous donation information is required for demographic analysis of retention.")
    
    with tab3:
        st.markdown('<div class="sub-header">Donor Engagement Analysis</div>', unsafe_allow_html=True)
        
        if has_previous_donation and 'Eligibility' in df.columns:
            # Create a retention/eligibility matrix
            st.markdown("### Donor Engagement Matrix")
            
            # Create contingency table
            engagement_matrix = pd.crosstab(
                df['Previous_Donation'],
                df['Eligibility']
            )
            
            # Calculate percentages
            total_donors = len(df)
            
            # Get counts for each quadrant
            loyal_donors = engagement_matrix.get('Eligible', {}).get('Yes', 0)
            potential_reactivation = engagement_matrix.get('Ineligible', {}).get('Yes', 0)
            new_qualified = engagement_matrix.get('Eligible', {}).get('No', 0)
            lost_opportunity = engagement_matrix.get('Ineligible', {}).get('No', 0)
            
            # Calculate percentages
            loyal_pct = (loyal_donors / total_donors) * 100 if total_donors > 0 else 0
            reactivation_pct = (potential_reactivation / total_donors) * 100 if total_donors > 0 else 0
            new_pct = (new_qualified / total_donors) * 100 if total_donors > 0 else 0
            lost_pct = (lost_opportunity / total_donors) * 100 if total_donors > 0 else 0
            
            # Create a 2x2 heatmap
            matrix_data = [
                [loyal_donors, potential_reactivation],
                [new_qualified, lost_opportunity]
            ]
            
            pct_data = [
                [loyal_pct, reactivation_pct],
                [new_pct, lost_pct]
            ]
            
            # Create a prettier matrix visualization
            fig = go.Figure()
            
            # Add cells
            annotations = []
            colorscale = [[0, 'rgb(255,255,255)'], [1, 'rgb(0,0,255)']]
            
            # Add cells with values
            fig.add_trace(go.Heatmap(
                z=matrix_data,
                x=['Eligible', 'Ineligible'],
                y=['Returning', 'New'],
                colorscale='Blues',
                showscale=False
            ))
            
            # Add annotations
            annotations = []
            for i, row in enumerate(['Returning', 'New']):
                for j, col in enumerate(['Eligible', 'Ineligible']):
                    annotations.append(dict(
                        x=col,
                        y=row,
                        text=f"<b>{matrix_data[i][j]}</b><br>({pct_data[i][j]:.1f}%)",
                        font=dict(color='black' if pct_data[i][j] < 50 else 'white', size=14),
                        showarrow=False
                    ))
            
            # Add segment labels
            annotations.append(dict(
                x='Eligible',
                y='Returning',
                text="LOYAL DONORS",
                font=dict(color='black', size=12),
                showarrow=False,
                yshift=20
            ))
            
            annotations.append(dict(
                x='Ineligible',
                y='Returning',
                text="POTENTIAL REACTIVATION",
                font=dict(color='black', size=12),
                showarrow=False,
                yshift=20
            ))
            
            annotations.append(dict(
                x='Eligible',
                y='New',
                text="NEW QUALIFIED",
                font=dict(color='black', size=12),
                showarrow=False,
                yshift=20
            ))
            
            annotations.append(dict(
                x='Ineligible',
                y='New',
                text="LOST OPPORTUNITY",
                font=dict(color='black', size=12),
                showarrow=False,
                yshift=20
            ))
            
            fig.update_layout(
                title="Donor Engagement Matrix",
                annotations=annotations,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            st.markdown("""
            ### Engagement Matrix Interpretation
            
            1. **Loyal Donors (Returning + Eligible)**: These are your most valuable donors who have donated before and remain eligible.
               - *Strategy*: Prioritize retention through recognition and incentives.
            
            2. **Potential Reactivation (Returning + Ineligible)**: These donors have donated before but are currently ineligible.
               - *Strategy*: Address health concerns and maintain contact for future eligibility.
            
            3. **New Qualified (New + Eligible)**: First-time donors who are eligible to donate.
               - *Strategy*: Focus on first-to-second donation conversion through immediate follow-up.
            
            4. **Lost Opportunity (New + Ineligible)**: First-time donors who are ineligible.
               - *Strategy*: Education on eligibility requirements and health improvements.
            """)
            
            # Key engagement metrics
            st.markdown("### Key Engagement Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            # Loyalty Rate
            loyalty_rate = (loyal_donors / returning_donors) * 100 if returning_donors > 0 else 0
            col1.metric(
                "Loyalty Rate",
                f"{loyalty_rate:.1f}%",
                help="Percentage of returning donors who are eligible to donate again"
            )
            
            # New Donor Qualification Rate
            qualification_rate = (new_qualified / new_donors) * 100 if new_donors > 0 else 0
            col2.metric(
                "New Donor Qualification",
                f"{qualification_rate:.1f}%",
                help="Percentage of new donors who are eligible to donate"
            )
            
            # Reactivation Potential
            reactivation_potential = (potential_reactivation / returning_donors) * 100 if returning_donors > 0 else 0
            col3.metric(
                "Reactivation Potential",
                f"{reactivation_potential:.1f}%",
                help="Percentage of returning donors who are currently ineligible but could potentially become eligible again"
            )
            
            # Donor value analysis
            st.markdown("### Donor Lifetime Value Analysis")
            
            # Create hypothetical donor value calculation
            avg_donations_per_year = 2.5  # Assumption
            avg_value_per_donation = 1  # Standardized unit (1 donation = 1 unit)
            avg_donor_lifespan = 5  # Years
            
            donor_lifetime_value = avg_donations_per_year * avg_value_per_donation * avg_donor_lifespan
            
            # Create a waterfall chart for donor value
            retention_impact = donor_lifetime_value * (retention_rate / 100)
            eligibility_impact = donor_lifetime_value * (loyalty_rate / 100)
            
            fig = go.Figure(go.Waterfall(
                name="Donor Value",
                orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Initial Value", "Retention Impact", "Eligibility Impact", "Adjusted Value"],
                textposition="outside",
                text=[f"{donor_lifetime_value:.1f}", f"{-retention_impact:.1f}", f"{-eligibility_impact:.1f}", f"{donor_lifetime_value - retention_impact - eligibility_impact:.1f}"],
                y=[donor_lifetime_value, -retention_impact, -eligibility_impact, 0],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title="Factors Affecting Donor Lifetime Value",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Donor Lifetime Value Calculation:**
            
            - Average donations per year: {avg_donations_per_year}
            - Standard value per donation: {avg_value_per_donation} unit
            - Average donor lifespan: {avg_donor_lifespan} years
            - Theoretical lifetime value: {donor_lifetime_value} units
            - Adjusted for retention: {donor_lifetime_value - retention_impact:.1f} units
            - Adjusted for eligibility: {donor_lifetime_value - retention_impact - eligibility_impact:.1f} units
            
            *Note: This is a simplified model for illustration purposes.*
            """)
        else:
            st.info("Both previous donation and eligibility information are required for engagement analysis.")
