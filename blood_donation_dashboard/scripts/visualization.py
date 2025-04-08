"""
Visualization Module for Blood Donation Dashboard

This module provides visualization functions for the blood donation dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def create_map_visualization(geo_df):
    """
    Create interactive map visualization of donor distribution.
    
    Parameters:
    -----------
    geo_df : pd.DataFrame
        Dataframe with geographical coordinates and donor counts
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive map figure
    """
    if geo_df is None or geo_df.empty:
        return None
    
    # Create scatter mapbox
    fig = px.scatter_mapbox(
        geo_df, 
        lat="lat", 
        lon="lon", 
        size="donor_count",
        color="donor_count",
        hover_name="donor_count",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15,
        zoom=10,
        title="Geographic Distribution of Blood Donors"
    )
    
    # Update layout
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Donor Count")
    )
    
    return fig

def create_donor_demographics(df):
    """
    Create demographic visualizations for donors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with donor information
        
    Returns:
    --------
    dict
        Dictionary of plotly figures for different demographics
    """
    if df is None or df.empty:
        return {}
    
    figures = {}
    
    # Age distribution
    if 'Age' in df.columns:
        fig_age = px.histogram(
            df, 
            x='Age', 
            nbins=20, 
            color_discrete_sequence=['#3366CC'],
            title='Age Distribution of Donors',
            labels={'Age': 'Age (years)', 'count': 'Number of Donors'}
        )
        fig_age.update_layout(bargap=0.1)
        figures['age_distribution'] = fig_age
    
    # Gender distribution
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig_gender = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender',
            title='Gender Distribution',
            color_discrete_sequence=px.colors.qualitative.Safe,
            hole=0.4
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label')
        figures['gender_distribution'] = fig_gender
    
    # Age group by gender
    if 'Age_Group' in df.columns and 'Gender' in df.columns:
        age_gender = pd.crosstab(df['Age_Group'], df['Gender'])
        
        fig_age_gender = px.bar(
            age_gender, 
            barmode='group',
            title='Age Groups by Gender',
            labels={'value': 'Number of Donors', 'Age_Group': 'Age Group'}
        )
        figures['age_gender_distribution'] = fig_age_gender
    
    # Education level
    education_cols = ['Niveau d\'etude', 'Niveau d\'étude', 'NIVEAU_D_ETUDE', 'Education', 'Niveau_d\'etude']
    edu_col = next((col for col in education_cols if col in df.columns), None)
    
    if edu_col:
        edu_counts = df[edu_col].value_counts().reset_index()
        edu_counts.columns = ['Education', 'Count']
        
        fig_edu = px.bar(
            edu_counts.sort_values('Count', ascending=False).head(10),
            x='Education',
            y='Count',
            title='Top 10 Education Levels',
            color_discrete_sequence=['#1E88E5']
        )
        figures['education_distribution'] = fig_edu
    
    # Profession
    profession_cols = ['Profession', 'PROFESSION', 'Métier']
    prof_col = next((col for col in profession_cols if col in df.columns), None)
    
    if prof_col:
        prof_counts = df[prof_col].value_counts().reset_index()
        prof_counts.columns = ['Profession', 'Count']
        
        fig_prof = px.bar(
            prof_counts.sort_values('Count', ascending=False).head(10),
            x='Profession',
            y='Count',
            title='Top 10 Professions',
            color_discrete_sequence=['#43A047']
        )
        figures['profession_distribution'] = fig_prof
    
    return figures

def create_health_conditions_vis(df):
    """
    Create visualizations for health conditions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with donor information
        
    Returns:
    --------
    dict
        Dictionary of plotly figures for health conditions
    """
    if df is None or df.empty:
        return {}
    
    figures = {}
    
    # Health conditions vs. eligibility
    health_cols = ['Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
                  'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
                  'Cardiaque', 'Tatoué', 'Scarifié']
    
    health_cols_present = [col for col in health_cols if col in df.columns]
    
    if health_cols_present and 'Eligibility' in df.columns:
        # Prepare data
        health_data = []
        
        for col in health_cols_present:
            # Cross-tabulate each health condition with eligibility
            cross_tab = pd.crosstab(df[col], df['Eligibility'], normalize='index') * 100
            
            if 'Eligible' in cross_tab.columns:
                health_data.append({
                    'Condition': col,
                    'Eligible_Percent': cross_tab['Eligible'].iloc[1] if 1 in cross_tab.index else 0,
                    'Condition_Count': df[col].sum()
                })
        
        health_df = pd.DataFrame(health_data)
        
        if not health_df.empty:
            # Bar chart of health conditions vs eligibility
            fig_health = px.bar(
                health_df.sort_values('Eligible_Percent'),
                x='Condition',
                y='Eligible_Percent',
                color='Condition_Count',
                title='Eligibility Rate by Health Condition',
                labels={
                    'Condition': 'Health Condition',
                    'Eligible_Percent': 'Eligible Percentage (%)',
                    'Condition_Count': 'Number of Cases'
                },
                color_continuous_scale=px.colors.sequential.Viridis
            )
            figures['health_eligibility'] = fig_health
    
    # Health risk score vs eligibility
    if 'Health_Risk_Score' in df.columns and 'Eligibility' in df.columns:
        risk_elig = pd.crosstab(df['Health_Risk_Score'], df['Eligibility'], normalize='index') * 100
        risk_elig_df = risk_elig.reset_index()
        
        if 'Eligible' in risk_elig_df.columns:
            fig_risk = px.line(
                risk_elig_df,
                x='Health_Risk_Score',
                y='Eligible',
                markers=True,
                title='Eligibility Rate by Health Risk Score',
                labels={
                    'Health_Risk_Score': 'Health Risk Score (Number of Conditions)',
                    'Eligible': 'Eligible Percentage (%)'
                }
            )
            figures['risk_eligibility'] = fig_risk
    
    return figures

def create_campaign_effectiveness_vis(df):
    """
    Create visualizations for campaign effectiveness analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with donor information
        
    Returns:
    --------
    dict
        Dictionary of plotly figures for campaign effectiveness
    """
    if df is None or df.empty:
        return {}
    
    figures = {}
    
    # Donation trends over time
    date_col = 'Date de remplissage de la fiche'
    if date_col in df.columns:
        # Ensure the column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Group by date and count donations
        donations_by_date = df[date_col].dt.date.value_counts().sort_index()
        donations_by_date = donations_by_date.reset_index()
        donations_by_date.columns = ['Date', 'Donations']
        
        fig_trend = px.line(
            donations_by_date,
            x='Date',
            y='Donations',
            title='Donation Trends Over Time',
            labels={'Date': 'Date', 'Donations': 'Number of Donations'}
        )
        figures['donation_trend'] = fig_trend
        
        # Donation trends by month
        if 'Donation_Month' in df.columns:
            donations_by_month = df['Donation_Month'].value_counts().sort_index()
            donations_by_month = donations_by_month.reset_index()
            donations_by_month.columns = ['Month', 'Donations']
            
            # Map month numbers to names
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            donations_by_month['Month_Name'] = donations_by_month['Month'].map(month_names)
            
            fig_month = px.bar(
                donations_by_month,
                x='Month_Name',
                y='Donations',
                title='Donations by Month',
                labels={'Month_Name': 'Month', 'Donations': 'Number of Donations'},
                color_discrete_sequence=['#FF7043']
            )
            figures['donation_month'] = fig_month
        
        # Blood type distribution analysis
        blood_type_col = next((col for col in df.columns if 'Groupe Sanguin' in col or 'Blood Type' in col or 'ABO' in col), None)
        if blood_type_col:
            blood_types = df[blood_type_col].value_counts()
            blood_types = blood_types.reset_index()
            blood_types.columns = ['Blood Type', 'Count']
            
            # Add Rhesus grouping
            blood_types['Rhesus'] = blood_types['Blood Type'].str.contains('\+').map({True: 'Positive', False: 'Negative'})
            blood_types['ABO Group'] = blood_types['Blood Type'].str.replace('\+|\-', '', regex=True)
            
            # Create blood type distribution chart
            fig_blood_type = px.bar(
                blood_types,
                x='Blood Type',
                y='Count',
                title='Blood Type Distribution',
                labels={'Blood Type': 'Blood Type', 'Count': 'Number of Donors'},
                color='Rhesus',
                color_discrete_map={
                    'Positive': '#E57373',  # Light red for positive
                    'Negative': '#64B5F6'   # Light blue for negative
                }
            )
            
            fig_blood_type.update_layout(
                xaxis_title='Blood Type',
                yaxis_title='Number of Donors'
            )
            
            figures['blood_type_distribution'] = fig_blood_type
            
            # Create ABO group distribution chart (simplified)
            abo_groups = blood_types.groupby('ABO Group')['Count'].sum().reset_index()
            
            fig_abo = px.pie(
                abo_groups,
                values='Count',
                names='ABO Group',
                title='ABO Blood Group Distribution',
                color='ABO Group',
                color_discrete_map={
                    'A': '#C62828',  # Dark red
                    'B': '#1976D2',  # Dark blue
                    'AB': '#6A1B9A',  # Purple
                    'O': '#2E7D32'   # Dark green
                }
            )
            
            fig_abo.update_layout(
                legend_title='ABO Group'
            )
            
            figures['abo_distribution'] = fig_abo
        
        # Phenotype analysis
        if 'Phenotype' in df.columns:
            # Extract most common phenotypes
            phenotypes = []
            for phenotype_str in df['Phenotype'].dropna():
                for p in str(phenotype_str).split(','):
                    p = p.strip()
                    if p:
                        phenotypes.append(p)
            
            phenotype_counts = pd.Series(phenotypes).value_counts().head(10)  # Top 10 phenotypes
            phenotype_df = phenotype_counts.reset_index()
            phenotype_df.columns = ['Phenotype', 'Count']
            
            # Create phenotype distribution chart
            fig_phenotype = px.bar(
                phenotype_df,
                x='Phenotype',
                y='Count',
                title='Top 10 Blood Phenotypes',
                labels={'Phenotype': 'Phenotype', 'Count': 'Number of Occurrences'},
                color='Phenotype'
            )
            
            fig_phenotype.update_layout(
                xaxis_title='Phenotype',
                yaxis_title='Number of Occurrences',
                xaxis={'categoryorder':'total descending'}
            )
            
            figures['phenotype_distribution'] = fig_phenotype
    
    # Demographic effectiveness analysis
    if 'Age_Group' in df.columns and 'Previous_Donation' in df.columns:
        age_prev_donation = pd.crosstab(
            df['Age_Group'], 
            df['Previous_Donation'],
            normalize='index'
        ) * 100
        
        if 'Yes' in age_prev_donation.columns:
            age_prev_donation = age_prev_donation.reset_index()
            
            fig_age_retention = px.bar(
                age_prev_donation,
                x='Age_Group',
                y='Yes',
                title='Previous Donation Rate by Age Group',
                labels={
                    'Age_Group': 'Age Group',
                    'Yes': 'Percentage of Repeat Donors (%)'
                },
                color_discrete_sequence=['#7CB342']
            )
            figures['age_retention'] = fig_age_retention
    
    if 'Gender' in df.columns and 'Previous_Donation' in df.columns:
        gender_prev_donation = pd.crosstab(
            df['Gender'], 
            df['Previous_Donation'],
            normalize='index'
        ) * 100
        
        if 'Yes' in gender_prev_donation.columns:
            gender_prev_donation = gender_prev_donation.reset_index()
            
            fig_gender_retention = px.bar(
                gender_prev_donation,
                x='Gender',
                y='Yes',
                title='Previous Donation Rate by Gender',
                labels={
                    'Gender': 'Gender',
                    'Yes': 'Percentage of Repeat Donors (%)'
                },
                color_discrete_sequence=['#26A69A']
            )
            figures['gender_retention'] = fig_gender_retention
    
    return figures

def create_donor_clusters_vis(df, cluster_results):
    """
    Create visualizations for donor clustering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe with donor information
    cluster_results : pd.DataFrame
        Dataframe with clustering results
        
    Returns:
    --------
    dict
        Dictionary of plotly figures for donor clusters
    """
    if df is None or df.empty or cluster_results is None or cluster_results.empty:
        return {}
    
    figures = {}
    
    # Ensure we have the cluster column
    if 'Cluster' not in cluster_results.columns:
        return {}
    
    # 3D scatter plot if we have PCA components
    pca_cols = [col for col in cluster_results.columns if col.startswith('PCA_')]
    if len(pca_cols) >= 3:
        fig_3d = px.scatter_3d(
            cluster_results,
            x=pca_cols[0],
            y=pca_cols[1],
            z=pca_cols[2],
            color='Cluster',
            title='Donor Clusters in 3D Space',
            labels={
                pca_cols[0]: 'Component 1',
                pca_cols[1]: 'Component 2',
                pca_cols[2]: 'Component 3',
                'Cluster': 'Cluster'
            }
        )
        figures['cluster_3d'] = fig_3d
    
    # 2D scatter plot if we have at least 2 PCA components
    elif len(pca_cols) >= 2:
        fig_2d = px.scatter(
            cluster_results,
            x=pca_cols[0],
            y=pca_cols[1],
            color='Cluster',
            title='Donor Clusters in 2D Space',
            labels={
                pca_cols[0]: 'Component 1',
                pca_cols[1]: 'Component 2',
                'Cluster': 'Cluster'
            }
        )
        figures['cluster_2d'] = fig_2d
    
    # Cluster profiles - mean values by cluster
    profile_cols = ['Age', 'Health_Risk_Score']
    
    # Add encoded columns that might be useful
    encoded_cols = [col for col in cluster_results.columns if '_Encoded' in col]
    profile_cols.extend(encoded_cols)
    
    # Filter to only columns that exist
    profile_cols = [col for col in profile_cols if col in cluster_results.columns]
    
    if profile_cols:
        # Calculate mean values for each feature by cluster
        cluster_profiles = cluster_results.groupby('Cluster')[profile_cols].mean().reset_index()
        
        # Melt the dataframe for easier plotting
        melted_profiles = pd.melt(
            cluster_profiles, 
            id_vars=['Cluster'],
            value_vars=profile_cols,
            var_name='Feature', 
            value_name='Average Value'
        )
        
        fig_profiles = px.bar(
            melted_profiles,
            x='Feature',
            y='Average Value',
            color='Cluster',
            barmode='group',
            title='Cluster Profiles: Average Feature Values by Cluster',
            labels={
                'Feature': 'Feature',
                'Average Value': 'Average Value',
                'Cluster': 'Cluster'
            }
        )
        figures['cluster_profiles'] = fig_profiles
    
    # Cluster distribution
    cluster_dist = cluster_results['Cluster'].value_counts().reset_index()
    cluster_dist.columns = ['Cluster', 'Count']
    
    fig_dist = px.pie(
        cluster_dist,
        values='Count',
        names='Cluster',
        title='Distribution of Donors Across Clusters',
        hole=0.4
    )
    fig_dist.update_traces(textposition='inside', textinfo='percent+label')
    figures['cluster_distribution'] = fig_dist
    
    return figures

def create_donor_retention_vis(df):
    """
    Create visualizations for donor retention analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing donor information with previous donation data
        
    Returns:
    --------
    dict
        Dictionary of plotly figures for retention visualizations
    """
    if 'Previous_Donation' not in df.columns:
        return None
    
    # Initialize results dictionary
    figs = {}
    
    # Create pie chart for returning vs new donors
    donation_counts = df['Previous_Donation'].value_counts()
    
    if 'Yes' in donation_counts and 'No' in donation_counts:
        fig_pie = px.pie(
            values=[donation_counts.get('Yes', 0), donation_counts.get('No', 0)],
            names=['Returning Donors', 'New Donors'],
            title='Returning vs New Donors',
            color_discrete_sequence=['#636EFA', '#EF553B'],
            hole=0.4
        )
        
        # Add percentage annotations
        total = donation_counts.sum()
        returning_pct = (donation_counts.get('Yes', 0) / total) * 100
        new_pct = (donation_counts.get('No', 0) / total) * 100
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}'
        )
        
        figs['donor_type_pie'] = fig_pie
    
    # Create donor retention by campaign type if available
    if 'Campaign_Type' in df.columns:
        # Calculate retention by campaign type
        retention_by_campaign = pd.crosstab(
            df['Campaign_Type'],
            df['Previous_Donation'],
            normalize='index'
        ) * 100
        
        retention_by_campaign = retention_by_campaign.reset_index()
        
        # Create bar chart
        if 'Yes' in retention_by_campaign.columns:
            fig_campaign = px.bar(
                retention_by_campaign,
                x='Campaign_Type',
                y='Yes',
                title='Retention Rate by Campaign Type',
                labels={
                    'Campaign_Type': 'Campaign Type',
                    'Yes': 'Retention Rate (%)'
                },
                color='Yes',
                text='Yes',
                color_continuous_scale='Blues'
            )
            
            fig_campaign.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            
            figs['retention_by_campaign'] = fig_campaign
    
    # Return all figures
    return figs

def create_prediction_vis(model_results, feature_importance=None):
    """
    Create visualizations for eligibility prediction model results.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing model metrics and results
    feature_importance : pd.DataFrame, optional
        DataFrame containing feature importance scores
        
    Returns:
    --------
    dict
        Dictionary of plotly figures for prediction visualizations
    """
    # Initialize results dictionary
    figs = {}
    
    # Create confusion matrix visualization
    if 'confusion_matrix' in model_results:
        cm = model_results['confusion_matrix']
        
        # Create confusion matrix heatmap
        z = cm
        x = ['Predicted Ineligible', 'Predicted Eligible']
        y = ['Actual Ineligible', 'Actual Eligible']
        
        # Change the text in the cells to reflect the values
        annotations = []
        for i, row in enumerate(z):
            for j, value in enumerate(row):
                annotations.append(dict(
                    x=x[j], 
                    y=y[i],
                    text=str(value),
                    font=dict(color='white' if value > z.max()/2 else 'black'),
                    showarrow=False
                ))
                
        fig_cm = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Blues',
            showscale=False
        ))
        
        fig_cm.update_layout(
            title='Confusion Matrix',
            annotations=annotations,
            xaxis=dict(side='top'),
            yaxis=dict(autorange='reversed')
        )
        
        figs['confusion_matrix'] = fig_cm
    
    # Create ROC curve visualization
    if 'fpr' in model_results and 'tpr' in model_results:
        fpr = model_results['fpr']
        tpr = model_results['tpr']
        auc = model_results.get('auc', 0)
        
        fig_roc = go.Figure()
        
        # Add ROC curve
        fig_roc.add_trace(go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add diagonal reference line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Reference Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig_roc.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis=dict(title='False Positive Rate'),
            yaxis=dict(title='True Positive Rate'),
            showlegend=True
        )
        
        figs['roc_curve'] = fig_roc
    
    # Create feature importance visualization
    if feature_importance is not None and not feature_importance.empty:
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            feature_importance.tail(10),  # Show top 10 features
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        figs['feature_importance'] = fig_importance
    
    # Return all figures
    return figs

# Function to export visualizations if needed
def export_figures(figures, output_dir='../dashboard/assets'):
    """
    Export figures to files for use in dashboard.
    
    Parameters:
    -----------
    figures : dict
        Dictionary of plotly figures
    output_dir : str
        Directory to save figures
        
    Returns:
    --------
    None
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export each figure as HTML
    for name, fig in figures.items():
        if fig is not None:
            fig.write_html(f"{output_dir}/{name}.html")
    
    print(f"Exported {len(figures)} figures to {output_dir}")
