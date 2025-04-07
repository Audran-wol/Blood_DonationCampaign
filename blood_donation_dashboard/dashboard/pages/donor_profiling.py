"""
Donor Profiling Page

This module displays clustering analysis to profile donors based on various features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from sklearn.decomposition import PCA

# Add parent directory to path to import from scripts folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from scripts.clustering import perform_clustering, analyze_clusters, prepare_data_for_clustering

def show_donor_profiling(df):
    """
    Display the donor profiling page using clustering techniques.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Filtered dataframe with donor information
    """
    # Header
    st.markdown('<div class="main-header">Donor Profiling via Clustering</div>', unsafe_allow_html=True)
    st.markdown("Analyze donor segments to identify ideal donor profiles and targeting strategies.")
    st.markdown("---")
    
    # Create tabs for different aspects of clustering analysis
    tab1, tab2, tab3 = st.tabs(["Cluster Analysis", "Donor Profiles", "Feature Importance"])
    
    with tab1:
        st.markdown('<div class="sub-header">Donor Clustering Analysis</div>', unsafe_allow_html=True)
        
        # Clustering parameters
        st.markdown("### Clustering Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Clustering algorithm selection
            algorithm = st.selectbox(
                "Clustering Algorithm",
                ["KMeans", "Agglomerative", "DBSCAN"],
                index=0
            )
            
            # Number of clusters (if applicable)
            if algorithm in ["KMeans", "Agglomerative"]:
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=10,
                    value=4
                )
            else:
                # For DBSCAN
                eps = st.slider(
                    "Epsilon (neighborhood size)",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1
                )
                min_samples = st.slider(
                    "Minimum Samples",
                    min_value=3,
                    max_value=20,
                    value=5
                )
        
        with col2:
            # Select features for clustering
            feature_categories = st.multiselect(
                "Feature Categories",
                ["Demographics", "Health Conditions", "Location", "Eligibility"],
                default=["Demographics", "Health Conditions"]
            )
            
            # Apply PCA
            use_pca = st.checkbox("Apply PCA for dimensionality reduction", value=True)
            
            if use_pca:
                n_components = st.slider(
                    "Number of PCA Components",
                    min_value=2,
                    max_value=10,
                    value=3
                )
        
        # Prepare clustering parameters
        clustering_params = {
            'algorithm': algorithm,
            'use_pca': use_pca
        }
        
        if algorithm in ["KMeans", "Agglomerative"]:
            clustering_params['n_clusters'] = n_clusters
        else:
            clustering_params['eps'] = eps
            clustering_params['min_samples'] = min_samples
        
        if use_pca:
            clustering_params['n_components'] = n_components
        
        # Feature selection based on categories
        demographic_cols = ['Age', 'Gender', 'Profession', 'Education', 'Age_Group']
        health_cols = [
            'Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
            'Drepanocytaire', 'Diabétique', 'Hypertendus', 'Asthmatiques',
            'Cardiaque', 'Tatoué', 'Scarifié', 'Health_Risk_Score'
        ]
        location_cols = ['District', 'Neighborhood', 'Latitude', 'Longitude']
        eligibility_cols = ['Eligibility', 'Previous_Donation']
        
        # Filter to only the columns that exist in our dataframe
        feature_cols = []
        
        if "Demographics" in feature_categories:
            feature_cols.extend([col for col in demographic_cols if col in df.columns])
        
        if "Health Conditions" in feature_categories:
            feature_cols.extend([col for col in health_cols if col in df.columns])
        
        if "Location" in feature_categories:
            feature_cols.extend([col for col in location_cols if col in df.columns])
        
        if "Eligibility" in feature_categories:
            feature_cols.extend([col for col in eligibility_cols if col in df.columns])
        
        # Button to perform clustering
        if st.button("Perform Clustering Analysis"):
            if feature_cols:
                with st.spinner("Performing clustering analysis..."):
                    try:
                        # Prepare data for clustering
                        X, feature_names = prepare_data_for_clustering(df, feature_cols)
                        
                        if X is not None and X.shape[0] > 10:  # Use shape[0] instead of len(X) for sparse matrices
                            # Perform clustering
                            clustering_result = perform_clustering(
                                df=df,  # Pass the original dataframe 
                                n_clusters=n_clusters
                            )
                            
                            if clustering_result is not None:
                                # Display cluster distribution
                                df_with_clusters = clustering_result
                                cluster_counts = df_with_clusters['Cluster'].value_counts().reset_index()
                                cluster_counts.columns = ['Cluster', 'Count']
                                
                                # Sort by cluster number
                                cluster_counts = cluster_counts.sort_values('Cluster')
                                
                                # Create bar chart for cluster distribution
                                fig_distribution = px.bar(
                                    cluster_counts,
                                    x='Cluster',
                                    y='Count',
                                    title='Cluster Size Distribution',
                                    color='Count',
                                    text='Count',
                                    color_continuous_scale='Viridis'
                                )
                                
                                fig_distribution.update_traces(texttemplate='%{text}', textposition='outside')
                                
                                st.plotly_chart(fig_distribution, use_container_width=True)
                                
                                # PCA Visualization (if available or calculated)
                                if hasattr(df_with_clusters, 'pca_result'):
                                    pca_data = df_with_clusters.pca_result
                                    
                                    if isinstance(pca_data, np.ndarray) and pca_data.shape[1] >= 2:
                                        # Create dataframe for PCA visualization
                                        pca_df = pd.DataFrame(
                                            pca_data[:, :2],
                                            columns=['PC1', 'PC2']
                                        )
                                        
                                        pca_df['Cluster'] = df_with_clusters['Cluster']
                                        
                                        # Create scatter plot for PCA visualization
                                        fig_pca = px.scatter(
                                            pca_df,
                                            x='PC1',
                                            y='PC2',
                                            color='Cluster',
                                            title='Cluster Visualization via PCA',
                                            hover_data=['Cluster']
                                        )
                                        
                                        st.plotly_chart(fig_pca, use_container_width=True)
                                
                                # Display cluster profiles if available
                                if hasattr(df_with_clusters, 'cluster_profiles'):
                                    st.markdown("### Cluster Profiles")
                                    st.dataframe(df_with_clusters.cluster_profiles)
                                
                                # Store the clustering result in session state for other tabs
                                st.session_state.clustering_result = clustering_result
                            else:
                                st.error("Clustering failed. Try different features or parameters.")
                        else:
                            st.warning("Not enough data for clustering. Select more features or filter criteria.")
                    except Exception as e:
                        st.error(f"An error occurred during clustering: {str(e)}")
                        st.info("Try selecting different features or adjusting the number of clusters.")
            else:
                st.warning("Please select at least one feature category for clustering.")
    
    with tab2:
        st.markdown('<div class="sub-header">Donor Profiles by Cluster</div>', unsafe_allow_html=True)
        
        # Check if clustering has been performed
        if 'clustering_result' in st.session_state:
            clustering_result = st.session_state.clustering_result
            df_with_clusters = clustering_result
            
            # Analyze clusters
            cluster_analysis = analyze_clusters(df_with_clusters, df)
            
            if cluster_analysis:
                # Select a cluster to analyze in detail
                clusters = sorted(df_with_clusters['Cluster'].unique())
                selected_cluster = st.selectbox(
                    "Select Cluster to Analyze",
                    clusters,
                    index=0
                )
                
                # Get profile for selected cluster
                if selected_cluster in cluster_analysis:
                    cluster_profile = cluster_analysis[selected_cluster]
                    
                    # Display basic cluster information
                    st.markdown(f"### Cluster {selected_cluster} Analysis")
                    
                    # Get cluster size
                    if 'cluster_sizes' in cluster_analysis:
                        cluster_size = cluster_analysis['cluster_sizes'].get(selected_cluster, 0)
                        st.metric("Cluster Size", f"{cluster_size} donors", f"{cluster_size / len(df) * 100:.1f}% of total")
                    
                    # Display cluster profile metrics
                    st.markdown("#### Key Metrics")
                    
                    # Create columns for metrics
                    cols = st.columns(3)
                    
                    # Add metrics to columns
                    metric_idx = 0
                    for key, value in cluster_profile.items():
                        if not key.startswith('PCA_') and not key == 'Cluster':
                            try:
                                # Format key for display
                                display_key = key.replace('_', ' ').title()
                                
                                # Format value based on type
                                if isinstance(value, (int, float)):
                                    if value > 1 and key.endswith('_mean') or key.endswith('_median'):
                                        display_value = f"{value:.1f}"
                                    elif value < 1:
                                        display_value = f"{value*100:.1f}%"
                                    else:
                                        display_value = f"{value:.1f}"
                                else:
                                    display_value = str(value)
                                
                                # Add to column
                                cols[metric_idx % 3].metric(display_key, display_value)
                                metric_idx += 1
                            except Exception as e:
                                print(f"Error displaying metric {key}: {e}")
                    
                    # Eligibility analysis if available
                    if 'Eligibility' in df.columns:
                        st.markdown("### Eligibility Analysis")
                        
                        # Filter for the selected cluster
                        cluster_df = df_with_clusters[df_with_clusters['Cluster'] == selected_cluster]
                        
                        if 'Eligibility' in cluster_df.columns:
                            # Calculate eligibility distribution
                            eligibility_counts = cluster_df['Eligibility'].value_counts(normalize=True) * 100
                            
                            # Calculate overall eligibility distribution for comparison
                            overall_eligibility = df['Eligibility'].value_counts(normalize=True) * 100
                            
                            # Create a comparison dataframe
                            compare_elig_df = pd.DataFrame({
                                'Status': list(eligibility_counts.index) + list(overall_eligibility.index),
                                'Percentage': list(eligibility_counts.values) + list(overall_eligibility.values),
                                'Group': ['This Cluster'] * len(eligibility_counts) + ['Overall'] * len(overall_eligibility)
                            })
                            
                            # Create a grouped bar chart
                            fig = px.bar(
                                compare_elig_df,
                                x='Status',
                                y='Percentage',
                                color='Group',
                                barmode='group',
                                title='Eligibility Comparison',
                                text='Percentage'
                            )
                            
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Show sample donors from this cluster
                    st.markdown("### Sample Donors in this Cluster")
                    
                    sample_donors = df_with_clusters[df_with_clusters['Cluster'] == selected_cluster].sample(
                        min(5, len(df_with_clusters[df_with_clusters['Cluster'] == selected_cluster]))
                    )
                    
                    # Select important columns to display
                    display_cols = []
                    for col_group in [demographic_cols, health_cols, eligibility_cols]:
                        display_cols.extend([col for col in col_group if col in sample_donors.columns])
                    
                    # Limit to a reasonable number of columns
                    if len(display_cols) > 10:
                        display_cols = display_cols[:10]
                    
                    # Add Cluster column
                    if 'Cluster' not in display_cols:
                        display_cols.append('Cluster')
                    
                    # Display the sample donors
                    st.dataframe(sample_donors[display_cols], use_container_width=True)
                else:
                    st.error(f"Cluster {selected_cluster} analysis not available.")
            else:
                st.info("Cluster analysis not available. Please run clustering first.")
        else:
            st.info("Please perform clustering analysis in the 'Cluster Analysis' tab first.")
    
    with tab3:
        st.markdown('<div class="sub-header">Feature Importance in Clustering</div>', unsafe_allow_html=True)
        
        # Check if clustering has been performed
        if 'clustering_result' in st.session_state:
            clustering_result = st.session_state.clustering_result
            
            # Extract feature importances if available
            if 'feature_importance' in clustering_result:
                feature_importance = clustering_result['feature_importance']
                
                if feature_importance is not None and len(feature_importance) > 0:
                    # Create a dataframe for visualization
                    fi_df = pd.DataFrame({
                        'Feature': list(feature_importance.keys()),
                        'Importance': list(feature_importance.values())
                    })
                    
                    # Sort by importance
                    fi_df = fi_df.sort_values('Importance', ascending=False)
                    
                    # Take top 15 features to avoid overcrowding
                    fi_df = fi_df.head(15)
                    
                    # Create a bar chart
                    fig = px.bar(
                        fi_df,
                        y='Feature',
                        x='Importance',
                        orientation='h',
                        title='Feature Importance in Clustering',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display feature importance table
                    st.markdown("### Feature Importance Table")
                    
                    # Add percentage column
                    fi_df['Percentage'] = fi_df['Importance'] / fi_df['Importance'].sum() * 100
                    
                    # Format the table
                    display_df = fi_df.copy()
                    display_df['Importance'] = display_df['Importance'].apply(lambda x: f"{x:.4f}")
                    display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Feature importance explanation
                    st.markdown("""
                    ### Understanding Feature Importance
                    
                    The importance scores above show how influential each feature is in defining the clusters. Features with higher scores are more important in distinguishing between different donor groups.
                    
                    #### Applications:
                    
                    - **Targeted Campaigns**: Focus on the top features to create more effective donor targeting
                    - **Questionnaire Optimization**: Prioritize collection of the most impactful information
                    - **Donor Segmentation**: Use the most important features for quick segmentation of new donors
                    """)
                else:
                    st.info("Feature importance not available for the selected clustering algorithm.")
            else:
                st.info("Feature importance not available for the selected clustering algorithm.")
            
            # Display cluster separation metrics if available
            if 'metrics' in clustering_result:
                metrics = clustering_result['metrics']
                
                st.markdown("### Clustering Quality Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                if 'silhouette' in metrics:
                    col1.metric(
                        "Silhouette Score",
                        f"{metrics['silhouette']:.3f}",
                        help="Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1]. Higher is better."
                    )
                
                if 'calinski_harabasz' in metrics:
                    col2.metric(
                        "Calinski-Harabasz Score",
                        f"{metrics['calinski_harabasz']:.1f}",
                        help="Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better."
                    )
                
                if 'davies_bouldin' in metrics:
                    col3.metric(
                        "Davies-Bouldin Index",
                        f"{metrics['davies_bouldin']:.3f}",
                        delta="-" if metrics['davies_bouldin'] < 0.8 else None,  # Lower is better
                        delta_color="normal" if metrics['davies_bouldin'] < 0.8 else "inverse",
                        help="Average similarity between clusters. Range: [0, ∞). Lower is better."
                    )
                
                # Add interpretation
                st.markdown("""
                #### Interpretation:
                
                - **Silhouette Score**: Values near 1 indicate well-separated clusters. Values near 0 indicate overlapping clusters.
                - **Calinski-Harabasz Score**: Higher values indicate better-defined clusters. No fixed range.
                - **Davies-Bouldin Index**: Lower values indicate better cluster separation. Typically values below 0.8 are considered good.
                """)
        else:
            st.info("Please perform clustering analysis in the 'Cluster Analysis' tab first.")
