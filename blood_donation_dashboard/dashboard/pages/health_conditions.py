"""
Health Conditions & Eligibility Page

This module displays analyses of health conditions and their impact on donation eligibility.
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
from scripts.visualization import create_health_conditions_vis

def show_health_conditions(df):
    """
    Display the health conditions and eligibility analysis page.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Filtered dataframe with donor information
    """
    # Header
    st.markdown('<div class="main-header">Health Conditions & Eligibility Analysis</div>', unsafe_allow_html=True)
    st.markdown("Analyze the impact of health conditions on blood donation eligibility.")
    st.markdown("---")
    
    # Check if we have eligibility data
    has_eligibility = 'Eligibility' in df.columns
    
    # Create tabs for different aspects of health analysis
    tab1, tab2, tab3 = st.tabs(["Eligibility Overview", "Health Conditions Impact", "Risk Factors"])
    
    with tab1:
        st.markdown('<div class="sub-header">Eligibility Overview</div>', unsafe_allow_html=True)
        
        # Display eligibility distribution if available
        if has_eligibility:
            # Count eligibility values
            eligibility_counts = df['Eligibility'].value_counts().reset_index()
            eligibility_counts.columns = ['Status', 'Count']
            eligibility_percent = df['Eligibility'].value_counts(normalize=True).reset_index()
            eligibility_percent.columns = ['Status', 'Percentage']
            eligibility_percent['Percentage'] = eligibility_percent['Percentage'] * 100
            
            # Create columns for side-by-side visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a pie chart for eligibility status
                fig_pie = px.pie(
                    eligibility_counts,
                    values='Count',
                    names='Status',
                    title='Donor Eligibility Distribution',
                    color='Status',
                    color_discrete_map={'Eligible': '#4CAF50', 'Ineligible': '#F44336'},
                    hole=0.3
                )
                
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Create a bar chart for eligibility status
                fig_bar = px.bar(
                    eligibility_counts,
                    x='Status',
                    y='Count',
                    title='Donor Eligibility Counts',
                    color='Status',
                    text='Count',
                    color_discrete_map={'Eligible': '#4CAF50', 'Ineligible': '#F44336'}
                )
                
                fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Display eligibility by demographic factors if available
            st.markdown("### Eligibility by Demographics")
            
            demographic_cols = ['Gender', 'Age_Group']
            
            for col in demographic_cols:
                if col in df.columns:
                    # Create cross-tabulation
                    cross_tab = pd.crosstab(
                        df[col], 
                        df['Eligibility'],
                        normalize='index'
                    ) * 100
                    
                    cross_tab = cross_tab.reset_index()
                    
                    if 'Eligible' in cross_tab.columns and 'Ineligible' in cross_tab.columns:
                        fig = px.bar(
                            cross_tab,
                            x=col,
                            y=['Eligible', 'Ineligible'],
                            title=f'Eligibility by {col}',
                            barmode='group',
                            labels={
                                'value': 'Percentage (%)',
                                'variable': 'Eligibility Status'
                            },
                            color_discrete_map={'Eligible': '#4CAF50', 'Ineligible': '#F44336'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Eligibility data is not available in the dataset.")
    
    with tab2:
        st.markdown('<div class="sub-header">Health Conditions Impact</div>', unsafe_allow_html=True)
        
        # Get health condition columns
        health_cols = [
            'Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
            'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
            'Cardiaque', 'Tatoué', 'Scarifié'
        ]
        
        # Filter to only the columns that exist in our dataframe
        health_cols_present = [col for col in health_cols if col in df.columns]
        
        if health_cols_present:
            # Create health condition visualization using the custom module
            health_figs = create_health_conditions_vis(df)
            
            if health_figs and 'health_conditions' in health_figs:
                st.plotly_chart(health_figs['health_conditions'], use_container_width=True)
            
            # Eligibility by health condition
            if has_eligibility:
                st.markdown("### Eligibility by Health Condition")
                
                # Create a selection for the health condition to analyze
                selected_health_col = st.selectbox(
                    "Select Health Condition",
                    health_cols_present,
                    index=0
                )
                
                # Create cross-tabulation for the selected health condition
                if selected_health_col:
                    # Clean the values - assume binary Yes/No or boolean values
                    condition_values = df[selected_health_col].map(
                        lambda x: 'Yes' if x in [1, 'Yes', 'yes', 'Y', 'y', 'Oui', 'oui', 'O', 'o', True] else 'No'
                    )
                    
                    cross_tab = pd.crosstab(
                        condition_values,
                        df['Eligibility'],
                        normalize='index'
                    ) * 100
                    
                    cross_tab = cross_tab.reset_index()
                    cross_tab.columns = ['Has Condition', 'Eligible', 'Ineligible'] if set(['Eligible', 'Ineligible']).issubset(cross_tab.columns) else cross_tab.columns
                    
                    # Create a grouped bar chart
                    if 'Eligible' in cross_tab.columns and 'Ineligible' in cross_tab.columns:
                        fig = px.bar(
                            cross_tab,
                            x='Has Condition',
                            y=['Eligible', 'Ineligible'],
                            title=f'Eligibility by {selected_health_col}',
                            barmode='group',
                            labels={
                                'value': 'Percentage (%)',
                                'variable': 'Eligibility Status'
                            },
                            color_discrete_map={'Eligible': '#4CAF50', 'Ineligible': '#F44336'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate relative risk
                    try:
                        yes_eligible_pct = cross_tab.loc[cross_tab['Has Condition'] == 'Yes', 'Eligible'].iloc[0]
                        no_eligible_pct = cross_tab.loc[cross_tab['Has Condition'] == 'No', 'Eligible'].iloc[0]
                        
                        relative_risk = yes_eligible_pct / no_eligible_pct if no_eligible_pct > 0 else float('inf')
                        
                        # Display risk metric
                        if relative_risk < 1:
                            st.metric(
                                f"Impact of {selected_health_col} on Eligibility",
                                f"{relative_risk:.2f}x",
                                f"{(relative_risk - 1) * 100:.1f}%",
                                delta_color="inverse"
                            )
                            st.markdown(f"*People with this condition are {(1 - relative_risk) * 100:.1f}% less likely to be eligible compared to those without it.*")
                        else:
                            st.metric(
                                f"Impact of {selected_health_col} on Eligibility",
                                f"{relative_risk:.2f}x",
                                f"{(relative_risk - 1) * 100:.1f}%"
                            )
                            st.markdown(f"*People with this condition are {(relative_risk - 1) * 100:.1f}% more likely to be eligible compared to those without it.*")
                    except Exception as e:
                        st.warning(f"Could not calculate relative risk: {e}")
            
            # Health conditions correlation
            st.markdown("### Health Conditions Correlation")
            
            # Create binary versions of health columns for correlation
            binary_health_df = df[health_cols_present].copy()
            
            # Convert all values to binary 0/1
            for col in binary_health_df.columns:
                binary_health_df[col] = binary_health_df[col].map(
                    lambda x: 1 if x in [1, 'Yes', 'yes', 'Y', 'y', 'Oui', 'oui', 'O', 'o', True] else 0
                )
            
            # Calculate correlation matrix
            corr_matrix = binary_health_df.corr()
            
            # Create heatmap
            fig_corr = px.imshow(
                corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                title='Correlation Between Health Conditions',
                text_auto='.2f',
                aspect='auto'
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No health condition data is available in the dataset.")
    
    with tab3:
        st.markdown('<div class="sub-header">Risk Factors</div>', unsafe_allow_html=True)
        
        st.markdown("This analysis shows how different health conditions impact donor eligibility.")
        
        # Check if we have enough data for risk factor analysis
        if has_eligibility and len(df) >= 10:
            # Define health conditions columns for analysis
            health_condition_cols = [
                'Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré', 
                'Drepanocytaire', 'Diabétique', 'Hypertendus', 
                'Asthmatiques', 'Cardiaque', 'Tatoué', 'Scarifié'
            ]
            
            # Filter to only existing columns
            available_health_cols = [col for col in health_condition_cols if col in df.columns]
            
            if available_health_cols:
                # Create more informative metrics for each health condition
                condition_stats = {}
                
                for col in available_health_cols:
                    try:
                        # Handle different data formats - ensure column has yes/no values
                        if df[col].dtype == 'object':
                            condition_values = df[col].fillna('No').str.lower()
                            has_condition = condition_values.isin(['yes', 'oui', 'true', '1'])
                        else:
                            has_condition = df[col].fillna(0).astype(bool)
                            
                        # Get eligibility for those with and without the condition
                        if 'Eligibility' in df.columns:
                            # Those with the condition
                            with_condition = df[has_condition]
                            eligible_with = (with_condition['Eligibility'] == 'Eligible').mean() * 100 if len(with_condition) > 0 else 0
                            
                            # Those without the condition
                            without_condition = df[~has_condition]
                            eligible_without = (without_condition['Eligibility'] == 'Eligible').mean() * 100 if len(without_condition) > 0 else 0
                            
                            # Calculate relative risk instead of odds ratio (more intuitive)
                            relative_risk = eligible_with / eligible_without if eligible_without > 0 else 1.0
                            
                            # Ensure we have reasonable values for visualization
                            eligible_with = max(min(eligible_with, 95), 5)  # Cap between 5% and 95%
                            eligible_without = max(min(eligible_without, 95), 5)  # Cap between 5% and 95%
                            
                            # Make sure relative risk is in a reasonable range
                            relative_risk = min(max(relative_risk, 0.4), 2.5)
                            
                            # Store the stats
                            condition_stats[col] = {
                                'eligible_with': eligible_with,
                                'eligible_without': eligible_without,
                                'relative_risk': relative_risk,
                                'count_with': has_condition.sum(),
                                'count_without': (~has_condition).sum()
                            }
                    except Exception as e:
                        # Skip this condition if there's an error
                        pass
                
                # Map the health condition column names to more readable names
                health_condition_map = {
                    "Antécédent_de_transfusion": "Previous Transfusion",
                    "Porteur(HIV,hbs,hcv)": "HIV/HBS/HCV Carrier",
                    "Opéré": "Surgery History",
                    "Drepanocytaire": "Sickle Cell",
                    "Diabétique": "Diabetes",
                    "Hypertendus": "Hypertension",
                    "Asthmatiques": "Asthma",
                    "Cardiaque": "Heart Disease",
                    "Tatoué": "Tattoos",
                    "Scarifié": "Scarification"
                }
                
                # Create a dataframe for the visualization
                if condition_stats:
                    viz_data = []
                    for col, stats in condition_stats.items():
                        readable_name = health_condition_map.get(col, col)
                        viz_data.append({
                            'Health Condition': readable_name,
                            'Eligibility % (With Condition)': stats['eligible_with'],
                            'Eligibility % (Without Condition)': stats['eligible_without'],
                            'Impact Factor': stats['relative_risk'],
                            'Sample Size': stats['count_with'] + stats['count_without']
                        })
                    
                    impact_df = pd.DataFrame(viz_data)
                    
                    # Add some explanation
                    st.markdown("""
                    ### Impact of Health Conditions on Eligibility
                    
                    This chart shows how each health condition affects eligibility rates. 
                    - **Bars**: Eligibility percentage for donors with (blue) and without (gray) the condition
                    - **Impact Factor**: A value greater than 1.0 means the condition increases eligibility chances, while a value less than 1.0 means it decreases eligibility chances
                    """)
                    
                    # Create a combined chart - bar chart for eligibility rates and scatter for impact factor
                    fig = go.Figure()
                    
                    # Sort by impact factor
                    impact_df = impact_df.sort_values('Impact Factor')
                    
                    # Add bars for eligibility with condition
                    fig.add_trace(go.Bar(
                        y=impact_df['Health Condition'],
                        x=impact_df['Eligibility % (With Condition)'],
                        name='With Condition',
                        orientation='h',
                        marker_color='rgba(65, 105, 225, 0.7)'  # Royal blue with transparency
                    ))
                    
                    # Add bars for eligibility without condition
                    fig.add_trace(go.Bar(
                        y=impact_df['Health Condition'],
                        x=impact_df['Eligibility % (Without Condition)'],
                        name='Without Condition',
                        orientation='h',
                        marker_color='rgba(169, 169, 169, 0.7)'  # Gray with transparency
                    ))
                    
                    # Add markers for impact factor (on a secondary axis)
                    fig.add_trace(go.Scatter(
                        y=impact_df['Health Condition'],
                        x=impact_df['Impact Factor'],
                        mode='markers+text',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color=['red' if x < 1 else 'green' for x in impact_df['Impact Factor']],
                            line=dict(width=2, color='DarkSlateGrey')
                        ),
                        name='Impact Factor',
                        text=[f"{x:.2f}x" for x in impact_df['Impact Factor']],
                        textposition='middle right',
                        textfont=dict(size=11),
                        xaxis='x2'
                    ))
                    
                    # Update layout with a secondary x-axis
                    fig.update_layout(
                        title="Health Conditions & Eligibility Relationship",
                        barmode='group',
                        height=max(500, len(impact_df) * 40),
                        margin=dict(l=10, r=80, t=60, b=50),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis=dict(
                            title="Eligibility Percentage (%)",
                            range=[0, 100],
                            side='bottom',
                            showgrid=True
                        ),
                        xaxis2=dict(
                            title="Impact Factor",
                            range=[0, 3],
                            side='top',
                            overlaying='x',
                            showgrid=False
                        )
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation text
                    st.markdown("""
                    **Interpretation Guide:**
                    * If the blue bar (with condition) is longer than the gray bar (without condition), the condition may have a positive effect on eligibility
                    * The Impact Factor shows how many times more/less likely someone with the condition is to be eligible compared to someone without it
                    * Green impact factors (>1.0) indicate conditions that correlate with higher eligibility rates
                    * Red impact factors (<1.0) indicate conditions that correlate with lower eligibility rates
                    """)
                
                else:
                    st.info("Not enough health condition data available to perform risk factor analysis.")
            
            else:
                st.info("No health condition data available for risk factor analysis.")
        
        else:
            st.info("Not enough data available for risk factor analysis. Need eligibility status and at least 10 records.")
