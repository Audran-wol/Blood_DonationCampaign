"""
Campaign Effectiveness Page

This module analyzes the effectiveness of blood donation campaigns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import calendar

# Add parent directory to path to import from scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from scripts.visualization import create_campaign_effectiveness_vis

def show_campaign_effectiveness(df):
    """
    Display the campaign effectiveness analysis page.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Filtered dataframe with donor information
    """
    # Header
    st.markdown('<div class="main-header">Campaign Effectiveness Analysis</div>', unsafe_allow_html=True)
    st.markdown("Analyze the effectiveness of blood donation campaigns over time and by different factors.")
    st.markdown("---")
    
    # Check if we have date information
    date_col = None
    date_cols = ['Date de remplissage de la fiche', 'Campaign_Date', 'Donation_Date']
    
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        st.warning("No date information is available in the dataset for time-based analysis.")
        has_dates = False
    else:
        # Ensure the date column is datetime type
        try:
            if not pd.api.types.is_datetime64_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            has_dates = True
        except Exception as e:
            st.error(f"Error converting dates: {e}")
            has_dates = False
    
    # Create tabs for different aspects of campaign effectiveness
    tab1, tab2, tab3 = st.tabs(["Temporal Analysis", "Campaign Comparisons", "Conversion Metrics"])
    
    with tab1:
        st.markdown('<div class="sub-header">Temporal Analysis of Campaigns</div>', unsafe_allow_html=True)
        
        if has_dates:
            # Create date filters
            col1, col2 = st.columns(2)
            
            with col1:
                min_date = df[date_col].min().date()
                max_date = df[date_col].max().date()
                
                start_date = st.date_input(
                    "Start Date",
                    min_date
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    max_date
                )
            
            # Apply date filter
            if start_date and end_date:
                # Convert to datetime for filtering
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.max.time())
                
                date_filtered_df = df[(df[date_col] >= start_datetime) & (df[date_col] <= end_datetime)]
                
                st.success(f"Showing data from {start_date} to {end_date} ({len(date_filtered_df)} records)")
            else:
                date_filtered_df = df
            
            # Create time-based visualizations
            time_figs = create_campaign_effectiveness_vis(date_filtered_df)
            
            if time_figs:
                # Monthly trend
                if 'donation_month' in time_figs:
                    st.plotly_chart(time_figs['donation_month'], use_container_width=True)
                
                # Yearly trend
                if 'donation_trend' in time_figs:
                    st.plotly_chart(time_figs['donation_trend'], use_container_width=True)
                
                # Daily pattern
                if 'donation_day' in time_figs:
                    st.plotly_chart(time_figs['donation_day'], use_container_width=True)
                
                # Blood type and phenotype analysis
                st.markdown("### Blood Type and Phenotype Analysis")
                
                bt_cols = st.columns(2)
                
                if 'blood_type_distribution' in time_figs:
                    bt_cols[0].plotly_chart(time_figs['blood_type_distribution'], use_container_width=True)
                
                if 'abo_distribution' in time_figs:
                    bt_cols[1].plotly_chart(time_figs['abo_distribution'], use_container_width=True)
                
                if 'phenotype_distribution' in time_figs:
                    st.plotly_chart(time_figs['phenotype_distribution'], use_container_width=True)
            
            # Time-based metrics
            st.markdown("### Key Time-Based Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            # Add peak donation period
            if date_col in date_filtered_df.columns:
                try:
                    # Monthly aggregation
                    date_filtered_df['Month'] = date_filtered_df[date_col].dt.month
                    date_filtered_df['Month_Name'] = date_filtered_df[date_col].dt.month_name()
                    
                    monthly_counts = date_filtered_df['Month'].value_counts().sort_index()
                    peak_month_idx = monthly_counts.idxmax()
                    peak_month = calendar.month_name[peak_month_idx]
                    
                    col1.metric(
                        "Peak Donation Month",
                        peak_month,
                        f"{monthly_counts[peak_month_idx]} donors"
                    )
                    
                    # Week day aggregation
                    date_filtered_df['Weekday'] = date_filtered_df[date_col].dt.day_name()
                    weekday_counts = date_filtered_df['Weekday'].value_counts()
                    peak_day = weekday_counts.idxmax()
                    
                    col2.metric(
                        "Peak Donation Day",
                        peak_day,
                        f"{weekday_counts[peak_day]} donors"
                    )
                    
                    # Year aggregation
                    if len(date_filtered_df[date_col].dt.year.unique()) > 1:
                        date_filtered_df['Year'] = date_filtered_df[date_col].dt.year
                        yearly_counts = date_filtered_df['Year'].value_counts().sort_index()
                        
                        # Calculate year-over-year growth
                        if len(yearly_counts) >= 2:
                            years = sorted(yearly_counts.index)
                            current_year = years[-1]
                            prev_year = years[-2]
                            
                            current_count = yearly_counts[current_year]
                            prev_count = yearly_counts[prev_year]
                            
                            growth = ((current_count - prev_count) / prev_count) * 100
                            
                            col3.metric(
                                "Year-over-Year Growth",
                                f"{growth:.1f}%",
                                growth
                            )
                        else:
                            col3.metric(
                                "Total Donations in Period",
                                f"{len(date_filtered_df)}",
                                None
                            )
                    else:
                        col3.metric(
                            "Total Donations in Period",
                            f"{len(date_filtered_df)}",
                            None
                        )
                    
                except Exception as e:
                    st.error(f"Error calculating time-based metrics: {e}")
            
        else:
            st.info("No date information is available for temporal analysis.")
    
    with tab2:
        st.markdown('<div class="sub-header">Campaign Comparison Analysis</div>', unsafe_allow_html=True)
        
        # Check if we have campaign/center information
        campaign_col = None
        campaign_cols = ['Campaign_Name', 'Centre', 'Campaign_ID', 'Center']
        
        for col in campaign_cols:
            if col in df.columns:
                campaign_col = col
                break
        
        if campaign_col is not None:
            # Create campaign/center selector
            campaigns = sorted(df[campaign_col].dropna().unique())
            
            if len(campaigns) > 1:
                selected_campaigns = st.multiselect(
                    f"Select {campaign_col}s to Compare",
                    campaigns,
                    default=campaigns[:min(5, len(campaigns))]
                )
                
                if selected_campaigns:
                    # Filter data for selected campaigns
                    campaign_df = df[df[campaign_col].isin(selected_campaigns)]
                    
                    # Create campaign comparison visualizations
                    st.markdown("### Campaign Performance Comparison")
                    
                    # Donor count by campaign
                    campaign_counts = campaign_df[campaign_col].value_counts().reset_index()
                    campaign_counts.columns = [campaign_col, 'Donor Count']
                    
                    fig_counts = px.bar(
                        campaign_counts,
                        x=campaign_col,
                        y='Donor Count',
                        title=f'Donor Count by {campaign_col}',
                        color='Donor Count',
                        text='Donor Count',
                        color_continuous_scale='Blues'
                    )
                    
                    fig_counts.update_traces(texttemplate='%{text}', textposition='outside')
                    
                    st.plotly_chart(fig_counts, use_container_width=True)
                    
                    # Eligibility rate by campaign
                    if 'Eligibility' in campaign_df.columns:
                        # Calculate eligibility percentage by campaign
                        elig_by_campaign = campaign_df.groupby(campaign_col)['Eligibility'].apply(
                            lambda x: (x == 'Eligible').mean() * 100
                        ).reset_index()
                        
                        elig_by_campaign.columns = [campaign_col, 'Eligibility Rate (%)']
                        
                        # Sort by eligibility rate
                        elig_by_campaign = elig_by_campaign.sort_values('Eligibility Rate (%)', ascending=False)
                        
                        fig_elig = px.bar(
                            elig_by_campaign,
                            x=campaign_col,
                            y='Eligibility Rate (%)',
                            title=f'Eligibility Rate by {campaign_col}',
                            color='Eligibility Rate (%)',
                            text='Eligibility Rate (%)',
                            color_continuous_scale='RdYlGn'
                        )
                        
                        fig_elig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        
                        # Add a target line at 80% eligibility (arbitrary good threshold)
                        fig_elig.add_shape(
                            type='line',
                            x0=-0.5,
                            y0=80,
                            x1=len(elig_by_campaign) - 0.5,
                            y1=80,
                            line=dict(color='red', width=2, dash='dash')
                        )
                        
                        st.plotly_chart(fig_elig, use_container_width=True)
                    
                    # Demographic comparison
                    st.markdown("### Demographic Comparison by Campaign")
                    
                    # Gender distribution by campaign
                    if 'Gender' in campaign_df.columns:
                        gender_campaign = pd.crosstab(
                            campaign_df[campaign_col],
                            campaign_df['Gender'],
                            normalize='index'
                        ) * 100
                        
                        gender_campaign = gender_campaign.reset_index()
                        
                        # Check if we have valid gender columns
                        if len(gender_campaign.columns) > 1:
                            fig_gender = px.bar(
                                gender_campaign,
                                x=campaign_col,
                                y=gender_campaign.columns[1:].tolist(),
                                title=f'Gender Distribution by {campaign_col}',
                                labels={
                                    'value': 'Percentage (%)',
                                    'variable': 'Gender'
                                },
                                barmode='stack'
                            )
                            
                            st.plotly_chart(fig_gender, use_container_width=True)
                    
                    # Age distribution by campaign
                    if 'Age' in campaign_df.columns:
                        fig_age = px.box(
                            campaign_df,
                            x=campaign_col,
                            y='Age',
                            title=f'Age Distribution by {campaign_col}',
                            color=campaign_col
                        )
                        
                        st.plotly_chart(fig_age, use_container_width=True)
                    
                    # Health condition comparison
                    st.markdown("### Health Condition Comparison by Campaign")
                    
                    # Get health condition columns
                    health_cols = [
                        'Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
                        'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                        'Cardiaque', 'Tatoué', 'Scarifié'
                    ]
                    
                    # Filter to only the columns that exist in our dataframe
                    health_cols_present = [col for col in health_cols if col in campaign_df.columns]
                    
                    if health_cols_present:
                        # Select a health condition to compare
                        selected_health_col = st.selectbox(
                            "Select Health Condition to Compare",
                            health_cols_present,
                            index=0
                        )
                        
                        # Calculate percentage of donors with the condition by campaign
                        health_by_campaign = campaign_df.groupby(campaign_col)[selected_health_col].apply(
                            lambda x: (x == 'Yes').mean() * 100 if x.dtype == 'object' else x.mean() * 100
                        ).reset_index()
                        
                        health_by_campaign.columns = [campaign_col, f'{selected_health_col} (%)']
                        
                        fig_health = px.bar(
                            health_by_campaign,
                            x=campaign_col,
                            y=f'{selected_health_col} (%)',
                            title=f'{selected_health_col} Prevalence by {campaign_col}',
                            color=f'{selected_health_col} (%)',
                            text=f'{selected_health_col} (%)',
                            color_continuous_scale='Oranges'
                        )
                        
                        fig_health.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        
                        st.plotly_chart(fig_health, use_container_width=True)
            else:
                st.info(f"Only one {campaign_col} is available in the dataset, so comparison is not possible.")
        else:
            st.info("No campaign/center information is available for campaign comparison.")
    
    with tab3:
        st.markdown('<div class="sub-header">Conversion Metrics</div>', unsafe_allow_html=True)
        
        # Check if we have the necessary data for conversion metrics
        has_previous_donation = 'Previous_Donation' in df.columns
        has_eligibility = 'Eligibility' in df.columns
        
        if has_previous_donation and has_eligibility:
            # Create a funnel chart for the donation pipeline
            st.markdown("### Donor Conversion Funnel")
            
            # Calculate funnel metrics
            total_donors = len(df)
            eligible_donors = df['Eligibility'].value_counts().get('Eligible', 0)
            previous_donors = df['Previous_Donation'].value_counts().get('Yes', 0)
            
            # New eligible donors (eligible and no previous donation)
            new_eligible = len(df[(df['Eligibility'] == 'Eligible') & (df['Previous_Donation'] == 'No')])
            
            # Returning eligible donors (eligible and previous donation)
            returning_eligible = len(df[(df['Eligibility'] == 'Eligible') & (df['Previous_Donation'] == 'Yes')])
            
            # Create funnel data
            funnel_data = pd.DataFrame({
                'Stage': ['Total Donors', 'Eligible Donors', 'New Eligible Donors', 'Returning Donors'],
                'Count': [total_donors, eligible_donors, new_eligible, returning_eligible]
            })
            
            # Create funnel chart
            fig_funnel = px.funnel(
                funnel_data,
                x='Count',
                y='Stage',
                title='Donor Conversion Funnel'
            )
            
            # Add percentage text
            percentages = [100]  # Total donors is 100%
            for i in range(1, len(funnel_data)):
                pct = (funnel_data['Count'][i] / funnel_data['Count'][0]) * 100
                percentages.append(pct)
            
            for i, (count, pct) in enumerate(zip(funnel_data['Count'], percentages)):
                fig_funnel.data[0].textinfo = 'value+percent previous'
            
            st.plotly_chart(fig_funnel, use_container_width=True)
            
            # Calculate conversion rates
            st.markdown("### Key Conversion Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            # Eligibility Rate
            eligibility_rate = (eligible_donors / total_donors) * 100 if total_donors > 0 else 0
            col1.metric(
                "Eligibility Rate",
                f"{eligibility_rate:.1f}%",
                help="Percentage of donors who are eligible to donate"
            )
            
            # Return Rate
            return_rate = (returning_eligible / eligible_donors) * 100 if eligible_donors > 0 else 0
            col2.metric(
                "Return Rate",
                f"{return_rate:.1f}%",
                help="Percentage of eligible donors who have donated before"
            )
            
            # Conversion Rate (new eligible donors as % of total new donors)
            new_donors = df['Previous_Donation'].value_counts().get('No', 0)
            conversion_rate = (new_eligible / new_donors) * 100 if new_donors > 0 else 0
            col3.metric(
                "New Donor Conversion",
                f"{conversion_rate:.1f}%",
                help="Percentage of new donors who are eligible to donate"
            )
            
            # Create a comparison of conversion metrics over time if date data is available
            if has_dates:
                st.markdown("### Conversion Metrics Over Time")
                
                try:
                    # Group by month
                    df['YearMonth'] = df[date_col].dt.to_period('M')
                    
                    # Calculate metrics by month
                    metrics_by_month = []
                    
                    for month, month_df in df.groupby('YearMonth'):
                        month_total = len(month_df)
                        month_eligible = month_df['Eligibility'].value_counts().get('Eligible', 0)
                        month_return = month_df['Previous_Donation'].value_counts().get('Yes', 0)
                        
                        month_elig_rate = (month_eligible / month_total) * 100 if month_total > 0 else 0
                        month_return_rate = (month_return / month_total) * 100 if month_total > 0 else 0
                        
                        metrics_by_month.append({
                            'Month': month,
                            'Eligibility Rate': month_elig_rate,
                            'Return Rate': month_return_rate,
                            'Donor Count': month_total
                        })
                    
                    # Convert to dataframe
                    metrics_df = pd.DataFrame(metrics_by_month)
                    metrics_df['Month'] = metrics_df['Month'].astype(str)
                    
                    # Create a dual-axis chart
                    fig = go.Figure()
                    
                    # Add eligibility rate line
                    fig.add_trace(go.Scatter(
                        x=metrics_df['Month'],
                        y=metrics_df['Eligibility Rate'],
                        name='Eligibility Rate (%)',
                        line=dict(color='green', width=2),
                        mode='lines+markers'
                    ))
                    
                    # Add return rate line
                    fig.add_trace(go.Scatter(
                        x=metrics_df['Month'],
                        y=metrics_df['Return Rate'],
                        name='Return Rate (%)',
                        line=dict(color='blue', width=2),
                        mode='lines+markers'
                    ))
                    
                    # Add donor count as bars on secondary y-axis
                    fig.add_trace(go.Bar(
                        x=metrics_df['Month'],
                        y=metrics_df['Donor Count'],
                        name='Donor Count',
                        marker_color='rgba(200, 200, 200, 0.6)',
                        yaxis='y2'
                    ))
                    
                    # Update layout for dual y-axes
                    fig.update_layout(
                        title='Conversion Metrics Trend',
                        xaxis=dict(title='Month'),
                        yaxis=dict(
                            title='Rate (%)',
                            range=[0, 100],
                            titlefont=dict(color='green'),
                            tickfont=dict(color='green')
                        ),
                        yaxis2=dict(
                            title='Donor Count',
                            titlefont=dict(color='gray'),
                            tickfont=dict(color='gray'),
                            anchor='x',
                            overlaying='y',
                            side='right'
                        ),
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating time-based conversion metrics: {e}")
        else:
            st.info("Either previous donation or eligibility data is missing, which is required for conversion metrics.")
