# Blood Donation Dashboard

A comprehensive dashboard application for analyzing blood donation data, identifying donor profiles, predicting eligibility, and optimizing blood donation campaigns.

## Overview

The Blood Donation Dashboard is an interactive web application built with Streamlit that provides healthcare professionals and blood bank administrators with a powerful tool for analyzing donor data, identifying patterns, and making data-driven decisions to improve blood donation campaigns and donor recruitment strategies.

## Features

### Data Exploration and Visualization
- **Donor Demographics**: Explore demographics of blood donors including age, gender, blood type, and location distributions.
- **Time-series Analysis**: Analyze donation trends over time with interactive charts and visualizations.
- **Filtering and Segmentation**: Filter and segment donor data based on various attributes.

### Donor Profiling with Clustering
- **Automated Clustering**: Identify natural donor segments using K-means clustering algorithm.
- **Profile Visualization**: Visualize cluster characteristics and distributions with interactive charts.
- **Optimal Cluster Selection**: Automatically determine the optimal number of clusters using silhouette scores.
- **PCA Visualization**: View clusters in 2D space using Principal Component Analysis.

### Eligibility Prediction
- **Machine Learning Models**: Predict donor eligibility using various machine learning algorithms.
- **Feature Importance**: Identify key factors affecting donation eligibility.
- **Model Evaluation**: Compare model performance with accuracy, precision, recall, and F1 scores.
- **Cross-validation**: Ensure model reliability through k-fold cross-validation.

### Campaign Effectiveness
- **Campaign Analysis**: Track and analyze the effectiveness of different recruitment campaigns.
- **Return Rate Analysis**: Measure donor return rates and retention by campaign.
- **Cost-Benefit Analysis**: Calculate the ROI and efficiency of various donor recruitment strategies.

## Technical Architecture

### Dashboard Components
- **Streamlit Frontend**: Interactive web interface built with Streamlit.
- **Plotly Visualizations**: Rich, interactive data visualizations using Plotly.
- **Pandas Data Processing**: Efficient data manipulation and analysis with Pandas.
- **Scikit-learn Machine Learning**: Predictive modeling and clustering algorithms.

### Code Structure
- `/dashboard/`: Contains the Streamlit application and UI components.
  - `app.py`: Main application entry point.
  - `/pages/`: Individual dashboard pages for different analysis features.
- `/scripts/`: Contains the backend processing and analysis modules.
  - `data_processing.py`: Functions for cleaning and preparing data.
  - `clustering.py`: Donor clustering and profiling algorithms.
  - `prediction_model.py`: Machine learning models for eligibility prediction.
  - `visualization.py`: Functions for generating visualizations.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python packages (install via requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blood-donation-dashboard.git
cd blood-donation-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
cd dashboard
streamlit run app.py
```

### Sample Data
The repository includes sample anonymized blood donation data for testing and demonstration purposes. For production use, connect to your own data sources by modifying the data loading functions in `data_processing.py`.

## Usage Examples

### Donor Profiling
1. Navigate to the "Donor Profiling" tab
2. Select feature categories for clustering (e.g., Demographics, Donation History)
3. Set the number of clusters or use "Auto" to find optimal clusters
4. Click "Perform Clustering Analysis" to view the results
5. Select individual clusters to analyze their characteristics

### Eligibility Prediction
1. Go to the "Eligibility Prediction" tab
2. Select features to include in the model
3. Choose a machine learning algorithm
4. Split the data for training and testing
5. Train the model and view performance metrics
6. Use the trained model to predict eligibility for new donors

## Contributing
Contributions to improve the Blood Donation Dashboard are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Thanks to all healthcare professionals and blood donation centers for their valuable feedback
- Special thanks to the open-source community for providing the tools that made this project possible
