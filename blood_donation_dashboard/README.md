# Blood Donation Dashboard

## Technical Documentation

This document provides detailed technical information about the Blood Donation Dashboard application, including architecture, implementation details, and developer guidelines.

## Project Structure

```
blood_donation_dashboard/
├── dashboard/                # Streamlit application
│   ├── app.py                # Main application entry point
│   ├── assets/               # Static assets (CSS, images)
│   └── pages/                # Dashboard page modules
│       ├── donor_profiling.py       # Donor clustering and profiling page
│       ├── eligibility_prediction.py # ML prediction page
│       ├── campaign_effectiveness.py # Campaign analysis page
│       └── data_explorer.py          # Data exploration page
├── scripts/                  # Core functionality modules
│   ├── clustering.py         # Clustering algorithms and donor profiling
│   ├── data_processing.py    # Data cleaning and preparation
│   ├── prediction_model.py   # ML models for eligibility prediction
│   └── visualization.py      # Data visualization functions
├── data/                     # Sample and processed data
├── models/                   # Saved ML models
├── tests/                    # Unit and integration tests
└── requirements.txt          # Python dependencies
```

## Core Modules

### 1. Data Processing (`scripts/data_processing.py`)

Responsible for loading, cleaning, and preparing data for analysis:

- `load_data()`: Loads blood donation data from CSV/Excel files
- `clean_data()`: Performs data cleaning operations (handling missing values, outliers)
- `prepare_features()`: Creates feature sets for machine learning models
- `process_categorical()`: Handles categorical data encoding

### 2. Clustering (`scripts/clustering.py`)

Implements donor segmentation and profiling algorithms:

- `prepare_data_for_clustering()`: Prepares data specifically for clustering
- `find_optimal_clusters()`: Determines the optimal number of clusters using silhouette scores
- `perform_clustering()`: Performs K-means clustering and returns results with PCA components
- `analyze_clusters()`: Generates insights from cluster data
- `profile_clusters()`: Creates descriptive profiles for each cluster

### 3. Prediction Model (`scripts/prediction_model.py`)

Implements eligibility prediction models:

- `prepare_data_for_modeling()`: Prepares features and target for ML models
- `train_model()`: Trains various ML models with cross-validation
- `evaluate_model()`: Calculates performance metrics for trained models
- `predict_eligibility()`: Makes predictions on new donor data
- `feature_importance()`: Calculates and visualizes feature importance

### 4. Visualization (`scripts/visualization.py`)

Provides visualization functions for dashboard components:

- `plot_demographics()`: Creates demographic distribution charts
- `plot_donation_trends()`: Creates time series visualizations
- `plot_cluster_profiles()`: Visualizes cluster characteristics
- `plot_feature_importance()`: Visualizes ML feature importance
- `plot_model_performance()`: Creates charts for model evaluation metrics

## Key Features Implementation Details

### Donor Clustering

The clustering functionality uses K-means algorithm with the following workflow:

1. Data preparation:
   - Numeric features are scaled using StandardScaler
   - Categorical features are encoded using one-hot encoding
   
2. Finding optimal clusters:
   - Uses silhouette score to evaluate cluster quality
   - Tries different values of k (typically 2-10)
   - Adapts to dataset size for better performance

3. PCA visualization:
   - Reduces dimensions to 2D for visualization
   - Preserves relationships between data points
   - Maps cluster assignments to colors

4. Cluster profiling:
   - Calculates statistics for each cluster (means, medians, modes)
   - Identifies distinguishing features between clusters
   - Generates human-readable cluster descriptions

### Eligibility Prediction

The eligibility prediction feature implements several machine learning algorithms:

1. Feature selection:
   - Correlation analysis
   - Feature importance from tree-based models
   - Domain-specific feature selection

2. Model training:
   - Supports multiple algorithms (Random Forest, LogisticRegression, etc.)
   - Hyperparameter tuning via grid search
   - Cross-validation to ensure model robustness

3. Model evaluation:
   - Calculates standard metrics (accuracy, precision, recall, F1)
   - Generates confusion matrices and ROC curves
   - Provides class imbalance handling

## Development Guidelines

### Adding New Features

1. Backend logic should be implemented in the appropriate script file
2. UI components should be added to the corresponding page file
3. Follow the existing patterns for data flow and state management
4. Include error handling and validation at all levels

### Testing

1. Unit tests should be added for all new functions
2. Integration tests should validate full workflows
3. Test with various data sizes and edge cases

### Performance Considerations

1. Large datasets:
   - Implement data sampling for initial visualizations
   - Use efficient algorithms for clustering (MiniBatchKMeans for very large data)
   - Consider caching for expensive computations

2. Streamlit optimization:
   - Use session state to persist data between interactions
   - Avoid unnecessary recomputation on UI interactions
   - Use st.cache_data for data loading functions

## Troubleshooting

### Common Issues

1. **Clustering Error: "TypeError: perform_clustering() got an unexpected keyword argument"**
   - Solution: Ensure all parameter names match between function calls and definitions
   - Check if the clustering.py module has been updated with the required parameters

2. **Memory Issues with Large Datasets**
   - Solution: Implement data sampling or chunking
   - Consider using sparse matrices for high-dimensional data after one-hot encoding

3. **PCA Visualization Error**
   - Solution: Verify that the input data has sufficient variance
   - Check if there are enough features (at least 2) for PCA reduction

4. **Model Training Performance**
   - Solution: Start with simpler models and gradually increase complexity
   - Use feature selection to reduce dimensionality

## Future Development Roadmap

1. Enhanced Clustering:
   - Implement additional clustering algorithms (DBSCAN, Hierarchical)
   - Add cluster stability analysis

2. Advanced Predictions:
   - Time-to-next-donation prediction
   - Donor churn prediction
   - Campaign response prediction

3. Dashboard Improvements:
   - Export functionality for analysis results
   - Custom reporting module
   - User management and permissions

4. Integration:
   - API endpoints for external system integration
   - Batch processing capabilities
   - Automated report generation
