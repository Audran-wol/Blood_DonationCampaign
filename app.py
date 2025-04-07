import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("blood_donation_data.csv")
    st.write("Available columns:", df.columns.tolist())  # Debug: Print columns
    return df

df = load_data()

# Rename columns if necessary
# Look at the data to see what the actual gender column is called
if 'Sexe' in df.columns:
    df['Gender'] = df['Sexe']
elif 'SEXE' in df.columns:
    df['Gender'] = df['SEXE']
elif 'sexe' in df.columns:
    df['Gender'] = df['sexe']

# Dashboard Header
st.title("Blood Donation Campaign Dashboard")
st.markdown("---")

# Key Metrics Section
st.header("Key Metrics")

# Map additional column names that might be different in the dataset
column_mappings = {
    # Map standard column names to possible alternatives in the dataset
    'ÉLIGIBILITÉ_AU_DON.': ['ÉLIGIBILITÉ_AU_DON.', 'Eligibilité', 'ELIGIBILITE', 'eligibilite', 'ÉLIGIBILITÉ'],
    'A-t-il_(elle)_déjà_donné_le_sang': ['A-t-il_(elle)_déjà_donné_le_sang', 'Déjà donné', 'DEJA_DONNE', 'Déjà_donné'],
    'Situation Matrimoniale': ['Situation Matrimoniale', 'SITUATION_MATRIMONIALE', 'Etat matrimonial'],
    'Niveau d\'étude': ['Niveau d\'étude', 'NIVEAU_D_ETUDE', 'Niveau d\'etude', 'Education'],
    'Religion': ['Religion', 'RELIGION'],
    'Profession': ['Profession', 'PROFESSION', 'Métier'],
    'Arrondissement de résidence': ['Arrondissement de résidence', 'ARRONDISSEMENT', 'Lieu de résidence']
}

# Helper function to find the actual column name in the dataframe
def get_actual_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

# Calculate metrics safely
total_donors = len(df)

# Get eligibility column and calculate metrics
eligibility_col = get_actual_column(column_mappings['ÉLIGIBILITÉ_AU_DON.'])
if eligibility_col:
    eligible_values = ['Eligible', 'eligible', 'OUI', 'Oui', 'oui', 'Yes', 'yes']
    eligible_count = sum(df[eligibility_col].isin(eligible_values))
    eligible_percent = (eligible_count / total_donors) * 100 if total_donors > 0 else 0
else:
    eligible_count = 0
    eligible_percent = 0
    st.warning("Eligibility column not found in dataset")

# Get gender metrics
if 'Gender' in df.columns:
    gender_dist = df['Gender'].value_counts(normalize=True) * 100
    male_values = ['Homme', 'homme', 'Male', 'male', 'M', 'HOMME']
    female_values = ['Femme', 'femme', 'Female', 'female', 'F', 'FEMME']
    
    male_percent = sum(df['Gender'].isin(male_values) / len(df) * 100) if len(df) > 0 else 0
    female_percent = sum(df['Gender'].isin(female_values) / len(df) * 100) if len(df) > 0 else 0
else:
    male_percent = 0
    female_percent = 0
    st.warning("Gender column not found in dataset")

# Get previous donation metrics
previous_donation_col = get_actual_column(column_mappings['A-t-il_(elle)_déjà_donné_le_sang'])
if previous_donation_col:
    no_values = ['No', 'NO', 'Non', 'NON', 'non', 'no']
    first_time_donors = sum(df[previous_donation_col].isin(no_values))
    repeat_donors = total_donors - first_time_donors
else:
    first_time_donors = 0
    repeat_donors = 0
    st.warning("Previous donation column not found in dataset")

# Create columns for metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Donors", f"{total_donors:,}")

with col2:
    st.metric("Eligibility Rate", f"{eligible_percent:.1f}%")

with col3:
    st.metric("Male Donors", f"{male_percent:.1f}%")

with col4:
    st.metric("Female Donors", f"{female_percent:.1f}%")

col5, col6, _ = st.columns(3)

with col5:
    st.metric("First-Time Donors", f"{first_time_donors:,}")

with col6:
    st.metric("Repeat Donors", f"{repeat_donors:,}")

st.markdown("---")

# Visualization of key metrics
st.subheader("Demographic Overview")

tab1, tab2, tab3, tab4 = st.tabs(["Eligibility", "Gender", "Age Distribution", "Previous Donation"])

with tab1:
    if eligibility_col:
        # Create a standardized eligibility series with consistent labeling
        eligibility_values = df[eligibility_col].copy()
        eligible_values = ['Eligible', 'eligible', 'OUI', 'Oui', 'oui', 'Yes', 'yes']
        eligibility_values = eligibility_values.apply(
            lambda x: 'Eligible' if str(x).strip() in eligible_values else 'Ineligible'
        )
        eligibility_counts = eligibility_values.value_counts()
        fig = px.pie(names=eligibility_counts.index, values=eligibility_counts.values, title='Eligibility Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Eligibility column not found")

with tab2:
    if 'Gender' in df.columns:
        # Clean gender values for consistency
        gender_values = df['Gender'].copy()
        gender_values = gender_values.apply(
            lambda x: 'Male' if str(x).strip() in ['Homme', 'homme', 'Male', 'male', 'M', 'HOMME'] 
            else ('Female' if str(x).strip() in ['Femme', 'femme', 'Female', 'female', 'F', 'FEMME'] else 'Other')
        )
        fig = px.pie(names=gender_values.value_counts().index, values=gender_values.value_counts().values, title='Gender Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Gender column not found")

with tab3:
    if 'Age' in df.columns:
        # Convert age to numeric, handling errors
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        fig = px.histogram(df, x='Age', nbins=20, title='Age Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Age column not found")

with tab4:
    previous_donation_col = get_actual_column(column_mappings['A-t-il_(elle)_déjà_donné_le_sang'])
    if previous_donation_col:
        # Clean previous donation values for consistency
        prev_donation_values = df[previous_donation_col].copy()
        yes_values = ['Yes', 'YES', 'Oui', 'OUI', 'oui', 'yes']
        no_values = ['No', 'NO', 'Non', 'NON', 'non', 'no']
        prev_donation_values = prev_donation_values.apply(
            lambda x: 'Yes' if str(x).strip() in yes_values 
            else ('No' if str(x).strip() in no_values else 'Unknown')
        )
        previous_donation_counts = prev_donation_values.value_counts()
        fig = px.pie(names=previous_donation_counts.index, values=previous_donation_counts.values, title='Previous Donation Status')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Previous donation column not found")

st.markdown("---")

st.subheader("Socio-Demographic Characteristics / Social and Personal Attributes")

tab5, tab6, tab7, tab8 = st.tabs(["Religion", "Education Level", "Marital Status", "Profession"])

with tab5:
    religion_col = get_actual_column(column_mappings['Religion'])
    if religion_col:
        religion_counts = df[religion_col].value_counts().head(10)
        fig = px.bar(x=religion_counts.index, y=religion_counts.values, title='Top 10 Religion Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Religion column not found")

with tab6:
    education_col = get_actual_column(column_mappings['Niveau d\'étude'])
    if education_col:
        education_counts = df[education_col].value_counts().head(10)
        fig = px.bar(x=education_counts.index, y=education_counts.values, title='Top 10 Education Level Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Education level column not found")

with tab7:
    marital_col = get_actual_column(column_mappings['Situation Matrimoniale'])
    if marital_col:
        marital_status_counts = df[marital_col].value_counts().head(10)
        fig = px.bar(x=marital_status_counts.index, y=marital_status_counts.values, title='Top 10 Marital Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Marital status column not found")

with tab8:
    profession_col = get_actual_column(column_mappings['Profession'])
    if profession_col:
        profession_counts = df[profession_col].value_counts().head(10)
        fig = px.bar(x=profession_counts.index, y=profession_counts.values, title='Top 10 Profession Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Profession column not found")

st.markdown("---")

# Health Condition Analysis
st.subheader("Reasons for Non-Eligibility")

health_conditions = [
    'Antécédent_de_transfusion', 'Porteur(HIV,hbs,hcv)', 'Opéré',
    'Drepanocytaire','Diabétique','Hypertendus', 'Asthmatiques',
    'Cardiaque', 'Tatoué','Scarifié'
]

condition_counts = {}
for condition in health_conditions:
    # Check if the condition column exists
    if condition in df.columns:
        # Look for Yes or similar values
        yes_values = ['Yes', 'YES', 'Oui', 'OUI', 'oui', 'yes']
        condition_counts[condition] = sum(df[condition].astype(str).str.strip().isin(yes_values))
    else:
        # Try common variants of the column name
        variants = [condition, condition.upper(), condition.lower(), 
                   condition.replace('é', 'e'), condition.replace('è', 'e')]
        found = False
        for variant in variants:
            if variant in df.columns:
                yes_values = ['Yes', 'YES', 'Oui', 'OUI', 'oui', 'yes']
                condition_counts[condition] = sum(df[variant].astype(str).str.strip().isin(yes_values))
                found = True
                break
        if not found:
            st.warning(f"Column '{condition}' not found in the dataset.")
            condition_counts[condition] = 0

non_eligibility_reasons = pd.Series(condition_counts).sort_values()
if any(non_eligibility_reasons > 0):
    plot_df = pd.DataFrame({
        'Non-Eligibility Reason': non_eligibility_reasons.index,
        'Number of Donors': non_eligibility_reasons.values
    })

    fig_non_eligible = px.bar(plot_df, x='Number of Donors', y='Non-Eligibility Reason',
                             orientation='h', title='Distribution of Non-Eligibility Reasons',
                             labels={'Number of Donors': 'Number of Donors', 'Non-Eligibility Reason': 'Reason'})
    st.plotly_chart(fig_non_eligible, use_container_width=True)
else:
    st.warning("No non-eligibility reasons found in the dataset.")

st.markdown("---")

st.subheader("Donor Profiling using Clustering")

# Define potential features based on column mappings
potential_features = [
    'Gender', 
    get_actual_column(column_mappings['Situation Matrimoniale']), 
    get_actual_column(column_mappings['Niveau d\'étude']), 
    get_actual_column(column_mappings['Profession']),
    get_actual_column(column_mappings['Arrondissement de résidence']), 
    get_actual_column(column_mappings['Religion']),
    get_actual_column(column_mappings['A-t-il_(elle)_déjà_donné_le_sang']),
    'Age'
]

# Add health conditions
potential_features.extend([col for col in health_conditions if col in df.columns])

# Filter out None values (columns that don't exist)
features = [feature for feature in potential_features if feature is not None]

if len(features) < 3:
    st.error("Not enough valid features found for clustering (minimum 3 needed).")
else:
    # Create a copy of the dataframe with only the available features
    cluster_df = df[features].copy()

    # Donor Profiling using Clustering
    # Handle Age column specially to ensure it remains numeric
    if 'Age' in cluster_df.columns:
        cluster_df['Age'] = pd.to_numeric(cluster_df['Age'], errors='coerce')
        # Fill missing age values with median
        cluster_df['Age'].fillna(cluster_df['Age'].median(), inplace=True)

    # Handle missing values in other columns
    for col in cluster_df.columns:
        if col != 'Age':  # Already handled Age above
            cluster_df[col].fillna('Unknown', inplace=True)

    # Identify binary features (health conditions)
    binary_features = [feature for feature in health_conditions if feature in features]
    
    # Convert binary features to numeric
    for feature in binary_features:
        yes_values = ['Yes', 'YES', 'Oui', 'OUI', 'oui', 'yes']
        cluster_df[feature] = cluster_df[feature].apply(
            lambda x: 1 if str(x).strip().lower() in [v.lower() for v in yes_values] else 0
        )

    # Separate numeric and categorical features
    numeric_features = ['Age'] if 'Age' in cluster_df.columns else []
    
    categorical_features = [col for col in cluster_df.columns 
                          if col != 'Age' and col not in binary_features]

    # Create preprocessor only if we have appropriate features
    if len(numeric_features) > 0 or len(categorical_features) > 0:
        # Define preprocessing pipeline
        transformers = []
        
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)

        # Determine optimal number of clusters using elbow method
        try:
            cluster_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('pca', PCA(n_components=0.95, svd_solver='full')),  # Reduce dimensions while preserving 95% variance
                ('cluster', KMeans(n_init=10, random_state=42))
            ])

            # Elbow method to find optimal k
            sse = []
            k_range = range(2, min(10, len(cluster_df) // 10 + 1))  # Ensure k is appropriate for dataset size
            
            for k in k_range:
                cluster_pipeline.set_params(cluster__n_clusters=k)
                cluster_pipeline.fit(cluster_df)
                sse.append(cluster_pipeline.named_steps['cluster'].inertia_)

            # Plot elbow method
            fig_elbow = px.line(x=list(k_range), y=sse, marker='o',
                              title='Elbow Method for Optimal k',
                              labels={'x': 'Number of clusters', 'y': 'SSE (Inertia)'})
            st.plotly_chart(fig_elbow)
            
            # Get the number of clusters from the slider
            n_clusters = st.slider("Number of Clusters for Profiling", 
                                min_value=2, 
                                max_value=min(9, len(cluster_df) // 10),
                                value=3)
            
            # Perform clustering with selected number of clusters
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pipeline_for_prediction = Pipeline(steps=[('preprocessor', preprocessor), ('kmeans', kmeans_model)])
            
            # Apply clustering and add to dataframe
            df['Cluster'] = pipeline_for_prediction.fit_predict(cluster_df)
            
            # Generate cluster summary with error handling
            try:
                agg_dict = {}
                if 'Age' in df.columns:
                    agg_dict['Age'] = ('Age', 'mean')
                
                if 'Gender' in df.columns:
                    agg_dict['Gender'] = ('Gender', lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
                
                profession_col = get_actual_column(column_mappings['Profession'])
                if profession_col:
                    agg_dict['Profession'] = (profession_col, lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
                
                religion_col = get_actual_column(column_mappings['Religion'])
                if religion_col:
                    agg_dict['Religion'] = (religion_col, lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
                
                eligibility_col = get_actual_column(column_mappings['ÉLIGIBILITÉ_AU_DON.'])
                if eligibility_col:
                    eligible_values = ['Eligible', 'eligible', 'OUI', 'Oui', 'oui', 'Yes', 'yes']
                    agg_dict['Eligibilité'] = (eligibility_col, lambda x: sum(x.isin(eligible_values)) / len(x) * 100 if len(x) > 0 else 0)
                
                previous_donation_col = get_actual_column(column_mappings['A-t-il_(elle)_déjà_donné_le_sang'])
                if previous_donation_col:
                    yes_values = ['Yes', 'YES', 'Oui', 'OUI', 'oui', 'yes']
                    agg_dict['Déjà_donné'] = (previous_donation_col, lambda x: sum(x.isin(yes_values)) / len(x) if len(x) > 0 else 0)
                
                # Add health conditions if available
                for condition in health_conditions:
                    if condition in df.columns:
                        yes_values = ['Yes', 'YES', 'Oui', 'OUI', 'oui', 'yes']
                        agg_dict[condition] = (condition, lambda x: sum(x.isin(yes_values)) / len(x) if len(x) > 0 else 0)
                
                # Always add count
                agg_dict['Count'] = ('Cluster', 'size')
                
                # Generate summary if we have at least some metrics
                if agg_dict:
                    cluster_summary = df.groupby('Cluster').agg(**agg_dict).reset_index()
                    st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))
                else:
                    st.warning("No valid metrics available for cluster summary.")
                
                # Visualize clusters using PCA
                try:
                    pca = PCA(n_components=2, random_state=42)
                    principal_components = pca.fit_transform(pipeline_for_prediction.named_steps['preprocessor'].transform(cluster_df))
                    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                    pca_df['Cluster'] = df['Cluster'].astype(str)
                    fig_cluster = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                                         title=f'Donor Clusters (PCA - {n_clusters} Clusters)')
                    st.plotly_chart(fig_cluster, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during cluster visualization: {e}")
                    
            except Exception as e:
                st.error(f"Error during cluster summary: {e}")
                
        except Exception as e:
            st.error(f"Error during clustering: {e}")
    else:
        st.error("No appropriate features available for clustering.")

st.markdown("---")

# Campaign Effectiveness Analysis
st.subheader("Campaign Effectiveness Analysis")

# Convert 'Date de remplissage' to datetime if it exists
date_columns = [col for col in df.columns if 'date' in col.lower() or 'remplissage' in col.lower()]
if date_columns:
    date_col = date_columns[0]  # Use the first matching date column
    
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Drop rows with missing dates for this analysis only
        date_df = df.dropna(subset=[date_col])
        
        if len(date_df) > 0:
            # Extract temporal features
            date_df['Donation_Year'] = date_df[date_col].dt.year
            date_df['Donation_Month'] = date_df[date_col].dt.month
            date_df['Month_Name'] = date_df[date_col].dt.strftime('%b')  # Abbreviated month name
            
            # Ensure months are ordered correctly
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Monthly Blood Donation Trends
            monthly_donations = date_df.groupby('Month_Name').size()
            
            # Sort by month order
            monthly_donations = monthly_donations.reindex(
                [m for m in month_order if m in monthly_donations.index]
            )
            
            fig_monthly = px.line(x=monthly_donations.index, y=monthly_donations.values,
                              markers=True, title='Monthly Blood Donation Trends')
            fig_monthly.update_layout(xaxis_title='Month', yaxis_title='Number of Donations')
            st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            st.warning(f"No valid dates found in the '{date_col}' column.")
    except Exception as e:
        st.error(f"Error processing date data: {e}")
else:
    st.warning("No date column found for temporal analysis.")

st.markdown("---")

# Donor Retention Analysis
st.subheader("Donor Retention Analysis")

# Find the column for previous donation status
previous_donation_col = get_actual_column(column_mappings['A-t-il_(elle)_déjà_donné_le_sang'])

if previous_donation_col:
    # Standardize values for visualization
    yes_values = ['Yes', 'YES', 'Oui', 'OUI', 'oui', 'yes']
    no_values = ['No', 'NO', 'Non', 'NON', 'non', 'no']
    
    # Create a standardized series
    previous_donation = df[previous_donation_col].apply(
        lambda x: 'Repeat Donor' if str(x).strip() in yes_values 
        else ('First-Time Donor' if str(x).strip() in no_values else 'Unknown')
    )
    
    # Calculate retention
    retention_by_previous = previous_donation.value_counts(normalize=True)
    
    # Filter out 'Unknown' for cleaner visualization
    if 'Unknown' in retention_by_previous.index:
        retention_by_previous = retention_by_previous.drop('Unknown')
    
    if not retention_by_previous.empty:
        fig_retention_pie = px.pie(
            names=retention_by_previous.index, 
            values=retention_by_previous.values,
            title='Donor Retention by Previous Donation Status'
        )
        st.plotly_chart(fig_retention_pie, use_container_width=True)
    else:
        st.warning("No valid data for donor retention analysis.")
else:
    st.warning("Previous donation status column not found in the dataset.")

st.markdown("---")

st.info("Blood Donation Campaign Dashboard")