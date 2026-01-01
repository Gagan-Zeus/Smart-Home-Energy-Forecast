# Energy Consumption Forecasting - Streamlit Web Application

## Overview
A comprehensive web application for analyzing and forecasting energy consumption patterns using PySpark and Machine Learning.

## Features

### 🏠 Home Page
- Project overview and objectives
- Quick statistics dashboard
- Dataset information summary

### 📊 Data Overview
- Dataset information and schema
- Sample data viewer
- Statistical summaries

### 📈 Data Analysis
- **Appliance Analysis**: Energy consumption by appliance type
- **Seasonal Analysis**: Energy patterns across seasons
- **Hourly Patterns**: Consumption trends throughout the day
- **Household Analysis**: Consumption by household size

### 🤖 Model Training
- Train Linear Regression and Random Forest models
- Real-time performance metrics (RMSE, MAE, R², Accuracy)
- Interactive model comparison charts
- Best model selection

### 🎯 Predictions
- Interactive prediction interface
- Input custom features
- Get energy consumption predictions
- Compare with historical averages

### 📉 Visualizations
- Correlation heatmaps
- Distribution analysis
- Time series plots
- 3D scatter plots

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Make sure your dataset file `smart_home_energy_consumption_large.csv` is in the same directory.

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Apache PySpark
- **Machine Learning**: PySpark MLlib
- **Visualizations**: Plotly, Matplotlib, Seaborn
- **Data Analysis**: Pandas, NumPy

## Features Highlights

### Interactive Dashboard
- Real-time data processing with Apache Spark
- Responsive design with multiple visualization options
- User-friendly navigation with sidebar menu

### Advanced Analytics
- 100,000+ records processing
- Multiple aggregation views
- Statistical analysis
- Pattern recognition

### Machine Learning
- Two regression models (Linear Regression & Random Forest)
- Automated feature engineering
- Model performance comparison
- Accuracy metrics

### Visualizations
- Interactive Plotly charts
- 3D visualizations
- Correlation analysis
- Time series plots

## Usage Guide

1. **Home**: Start here to see project overview and quick stats
2. **Data Overview**: Explore the dataset structure and samples
3. **Data Analysis**: Analyze patterns by appliance, season, hour, or household
4. **Model Training**: Click "Train Models" to build ML models
5. **Predictions**: Use trained models to predict energy consumption
6. **Visualizations**: Explore advanced visual analytics

## Performance Notes

- First load may take a few seconds to initialize Spark
- Model training can take 2-5 minutes depending on system resources
- Data is cached for faster subsequent loads
- Use sampling for large-scale visualizations

## Troubleshooting

### Common Issues

1. **Data file not found**: Ensure `smart_home_energy_consumption_large.csv` is in the project directory

2. **Memory error**: Increase Spark memory in `app.py`:
   ```python
   .config("spark.driver.memory", "8g")
   .config("spark.executor.memory", "8g")
   ```

3. **Port already in use**: Run with custom port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

## Project Structure

```
BDA Mini Project/
├── app.py                                      # Main Streamlit application
├── requirements.txt                            # Python dependencies
├── README_APP.md                               # This file
├── smart_home_energy_consumption_large.csv    # Dataset
└── Energy_Consumption_Forecasting.ipynb       # Jupyter notebook (optional)
```

## Credits

BDA Lab Mini Project - Energy Consumption Forecasting
Powered by PySpark, Streamlit, and Machine Learning
