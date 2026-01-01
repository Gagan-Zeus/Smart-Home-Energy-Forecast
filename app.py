import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, DecisionTreeRegressor, RandomForestRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Spark Session
@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("Energy Consumption Forecasting") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    return spark

# Load data - returns Spark DataFrame (not cached)
def load_data():
    spark = init_spark()
    df = spark.read.csv(
        "smart_home_energy_consumption_large.csv",
        header=True,
        inferSchema=True
    )
    return df

# Process data - returns Spark DataFrame (not cached)
def process_data(_df):
    # Clean data
    df_cleaned = _df.dropna().dropDuplicates()

    # Rename columns
    df_cleaned = df_cleaned \
        .withColumnRenamed("Home ID", "home_id") \
        .withColumnRenamed("Appliance Type", "appliance_type") \
        .withColumnRenamed("Energy Consumption (kWh)", "energy_consumption") \
        .withColumnRenamed("Time", "time") \
        .withColumnRenamed("Date", "date") \
        .withColumnRenamed("Outdoor Temperature (°C)", "outdoor_temperature") \
        .withColumnRenamed("Season", "season") \
        .withColumnRenamed("Household Size", "household_size")

    # Extract time features
    df_cleaned = df_cleaned.withColumn(
        "hour",
        when(length(trim(col("time"))) > 8,
             split(trim(col("time")), " ")[1])
        .otherwise(trim(col("time")))
    )

    df_cleaned = df_cleaned.withColumn(
        "hour",
        split(col("hour"), ":")[0].cast("integer")
    )

    df_cleaned = df_cleaned.withColumn(
        "date_parsed",
        to_date(col("date"), "yyyy-MM-dd")
    )

    df_cleaned = df_cleaned \
        .withColumn("month", month(col("date_parsed"))) \
        .withColumn("year", year(col("date_parsed"))) \
        .withColumn("day_of_week", dayofweek(col("date_parsed")))

    # Filter outliers
    df_filtered = df_cleaned.filter(
        (col("energy_consumption") > 0) &
        (col("energy_consumption") < 20)
    )

    return df_filtered

def train_models(_df_filtered):
    # Prepare ML data
    ml_data = _df_filtered.select(
        "energy_consumption",
        "appliance_type",
        "outdoor_temperature",
        "season",
        "household_size",
        "hour",
        "month",
        "day_of_week"
    )

    # Encode categorical variables
    appliance_indexer = StringIndexer(inputCol="appliance_type", outputCol="appliance_index", handleInvalid="keep")
    season_indexer = StringIndexer(inputCol="season", outputCol="season_index", handleInvalid="keep")

    # Fit and store the indexers
    appliance_indexer_model = appliance_indexer.fit(ml_data)
    season_indexer_model = season_indexer.fit(ml_data)

    ml_data = appliance_indexer_model.transform(ml_data)
    ml_data = season_indexer_model.transform(ml_data)

    # Create simple interaction features
    ml_data = ml_data.withColumn("temp_squared", col("outdoor_temperature") * col("outdoor_temperature"))

    # Assemble features
    feature_columns = [
        "appliance_index", "outdoor_temperature", "season_index",
        "household_size", "hour", "month", "day_of_week", "temp_squared"
    ]

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="unscaled_features")
    ml_data = assembler.transform(ml_data)

    # Scale features
    scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=False)
    scaler_model = scaler.fit(ml_data)
    ml_data = scaler_model.transform(ml_data)

    final_data = ml_data.select("features", col("energy_consumption").alias("label"))

    # Split data
    train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

    # Train Gradient Boosted Trees
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxIter=50,
        maxDepth=8,
        seed=42
    )
    gbt_model = gbt.fit(train_data)
    gbt_predictions = gbt_model.transform(test_data)

    # Train Decision Tree
    dt = DecisionTreeRegressor(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxDepth=15,
        minInstancesPerNode=5,
        seed=42
    )
    dt_model = dt.fit(train_data)
    dt_predictions = dt_model.transform(test_data)

    # Train Random Forest
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        numTrees=50,
        maxDepth=12,
        minInstancesPerNode=5,
        seed=42
    )
    rf_model = rf.fit(train_data)
    rf_predictions = rf_model.transform(test_data)

    # Evaluate models
    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    gbt_metrics = {
        'rmse': evaluator_rmse.evaluate(gbt_predictions),
        'mae': evaluator_mae.evaluate(gbt_predictions),
        'r2': evaluator_r2.evaluate(gbt_predictions)
    }

    dt_metrics = {
        'rmse': evaluator_rmse.evaluate(dt_predictions),
        'mae': evaluator_mae.evaluate(dt_predictions),
        'r2': evaluator_r2.evaluate(dt_predictions)
    }

    rf_metrics = {
        'rmse': evaluator_rmse.evaluate(rf_predictions),
        'mae': evaluator_mae.evaluate(rf_predictions),
        'r2': evaluator_r2.evaluate(rf_predictions)
    }

    return gbt_model, dt_model, rf_model, gbt_metrics, dt_metrics, rf_metrics, gbt_predictions, dt_predictions, rf_predictions, appliance_indexer_model, season_indexer_model, assembler, scaler_model

# Main App
def main():
    st.title("Energy Consumption Forecasting System")
    st.markdown("### BDA Lab Mini Project - Smart Home Energy Analysis")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["Home", "Data Overview", "Data Analysis", "Model Training", "Predictions", "Visualizations"]
    )

    # Load data
    try:
        with st.spinner("Loading data..."):
            df = load_data()
            df_filtered = process_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure 'smart_home_energy_consumption_large.csv' is in the same directory.")
        return

    # HOME PAGE
    if page == "Home":
        st.header("Welcome to Energy Consumption Forecasting Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Project Overview
            This application analyzes smart meter data using PySpark to predict energy consumption patterns.

            **Key Features:**
            - Real-time data processing with Apache Spark
            - Advanced machine learning models
            - Interactive visualizations
            - Comprehensive energy analytics

            **Problem Statement:**
            Utility companies must predict demand to balance supply efficiently.

            **Solution:**
            ML-based forecasting models for accurate energy prediction.
            """)

        with col2:
            st.markdown("""
            ### Objectives
            - Analyze 100,000+ smart meter records
            - Identify consumption patterns
            - Train predictive models (GBT, Decision Tree & Random Forest)
            - Provide actionable insights

            ### Dataset Features
            - Home ID
            - Appliance Type
            - Energy Consumption (kWh)
            - Outdoor Temperature
            - Season
            - Household Size
            - Time & Date
            """)

        # Quick Stats
        st.markdown("---")
        st.subheader("Quick Statistics")

        total_records = df_filtered.count()
        df_pd = df_filtered.select("energy_consumption", "appliance_type", "household_size").toPandas()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{total_records:,}")
        with col2:
            st.metric("Avg Consumption", f"{df_pd['energy_consumption'].mean():.2f} kWh")
        with col3:
            st.metric("Unique Appliances", df_pd['appliance_type'].nunique())
        with col4:
            st.metric("Avg Household Size", f"{df_pd['household_size'].mean():.1f}")

    # DATA OVERVIEW PAGE
    elif page == "Data Overview":
        st.header("Data Overview")

        tab1, tab2, tab3 = st.tabs(["Dataset Info", "Sample Data", "Statistics"])

        with tab1:
            st.subheader("Dataset Information")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Records", f"{df_filtered.count():,}")
                st.metric("Number of Columns", len(df_filtered.columns))

            with col2:
                st.write("**Column Names:**")
                st.write(df_filtered.columns)

            st.subheader("Data Schema")
            schema_df = pd.DataFrame([(field.name, str(field.dataType)) for field in df_filtered.schema.fields],
                                    columns=['Column', 'Data Type'])
            st.dataframe(schema_df, use_container_width=True)

        with tab2:
            st.subheader("Sample Data")
            sample_size = st.slider("Number of rows to display:", 5, 50, 10)
            sample_data = df_filtered.limit(sample_size).toPandas()
            st.dataframe(sample_data, use_container_width=True)

        with tab3:
            st.subheader("Statistical Summary")
            stats = df_filtered.select("energy_consumption", "outdoor_temperature", "household_size").describe().toPandas()
            st.dataframe(stats, use_container_width=True)

    # DATA ANALYSIS PAGE
    elif page == "Data Analysis":
        st.header("Data Analysis")

        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Appliance Analysis", "Seasonal Analysis", "Hourly Patterns", "Household Analysis"]
        )

        if analysis_type == "Appliance Analysis":
            st.subheader("Energy Consumption by Appliance Type")

            appliance_stats = df_filtered.groupBy("appliance_type").agg(
                count("energy_consumption").alias("count"),
                avg("energy_consumption").alias("avg_consumption"),
                sum("energy_consumption").alias("total_consumption")
            ).orderBy(col("total_consumption").desc()).toPandas()

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(appliance_stats, use_container_width=True)

            with col2:
                fig = px.bar(appliance_stats, x='appliance_type', y='avg_consumption',
                           title='Average Energy Consumption by Appliance',
                           labels={'avg_consumption': 'Avg Consumption (kWh)', 'appliance_type': 'Appliance Type'},
                           color='avg_consumption',
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Seasonal Analysis":
            st.subheader("Seasonal Energy Patterns")

            season_stats = df_filtered.groupBy("season").agg(
                count("*").alias("count"),
                avg("energy_consumption").alias("avg_consumption"),
                avg("outdoor_temperature").alias("avg_temperature")
            ).orderBy(col("avg_consumption").desc()).toPandas()

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(season_stats, use_container_width=True)

            with col2:
                fig = px.pie(season_stats, values='count', names='season',
                           title='Data Distribution by Season',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Hourly Patterns":
            st.subheader("Hourly Energy Consumption Patterns")

            hourly_stats = df_filtered.groupBy("hour").agg(
                avg("energy_consumption").alias("avg_consumption")
            ).orderBy("hour").toPandas()

            fig = px.line(hourly_stats, x='hour', y='avg_consumption',
                        title='Average Energy Consumption Throughout the Day',
                        labels={'avg_consumption': 'Avg Consumption (kWh)', 'hour': 'Hour of Day'},
                        markers=True)
            fig.update_traces(line_color='#1f77b4', line_width=3)
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Household Analysis":
            st.subheader("Energy Consumption by Household Size")

            household_stats = df_filtered.groupBy("household_size").agg(
                count("*").alias("count"),
                avg("energy_consumption").alias("avg_consumption")
            ).orderBy("household_size").toPandas()

            fig = px.bar(household_stats, x='household_size', y='avg_consumption',
                       title='Average Consumption vs Household Size',
                       labels={'avg_consumption': 'Avg Consumption (kWh)', 'household_size': 'Household Size'},
                       color='avg_consumption',
                       color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

    # MODEL TRAINING PAGE
    elif page == "Model Training":
        st.header("Model Training & Evaluation")

        if st.button("Train Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes..."):
                try:
                    gbt_model, dt_model, rf_model, gbt_metrics, dt_metrics, rf_metrics, gbt_pred, dt_pred, rf_pred, app_indexer, seas_indexer, feat_assembler, feat_scaler = train_models(df_filtered)

                    st.success("Models trained successfully!")

                    # Store in session state
                    st.session_state['gbt_model'] = gbt_model
                    st.session_state['dt_model'] = dt_model
                    st.session_state['rf_model'] = rf_model
                    st.session_state['gbt_metrics'] = gbt_metrics
                    st.session_state['dt_metrics'] = dt_metrics
                    st.session_state['rf_metrics'] = rf_metrics
                    st.session_state['gbt_pred'] = gbt_pred
                    st.session_state['dt_pred'] = dt_pred
                    st.session_state['rf_pred'] = rf_pred
                    st.session_state['appliance_indexer'] = app_indexer
                    st.session_state['season_indexer'] = seas_indexer
                    st.session_state['assembler'] = feat_assembler
                    st.session_state['scaler'] = feat_scaler
                    st.session_state['training_data'] = df_filtered

                except Exception as e:
                    st.error(f"Error training models: {e}")
                    return

        if 'gbt_metrics' in st.session_state:
            st.markdown("---")
            st.subheader("Model Performance Comparison")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Gradient Boosted Trees")
                st.metric("RMSE", f"{st.session_state['gbt_metrics']['rmse']:.4f}")
                st.metric("MAE", f"{st.session_state['gbt_metrics']['mae']:.4f}")
                st.metric("R² Score", f"{st.session_state['gbt_metrics']['r2']:.4f}")
                st.metric("Accuracy", f"{st.session_state['gbt_metrics']['r2']*100:.2f}%")

            with col2:
                st.markdown("#### Decision Tree")
                st.metric("RMSE", f"{st.session_state['dt_metrics']['rmse']:.4f}")
                st.metric("MAE", f"{st.session_state['dt_metrics']['mae']:.4f}")
                st.metric("R² Score", f"{st.session_state['dt_metrics']['r2']:.4f}")
                st.metric("Accuracy", f"{st.session_state['dt_metrics']['r2']*100:.2f}%")

            with col3:
                st.markdown("#### Random Forest")
                st.metric("RMSE", f"{st.session_state['rf_metrics']['rmse']:.4f}")
                st.metric("MAE", f"{st.session_state['rf_metrics']['mae']:.4f}")
                st.metric("R² Score", f"{st.session_state['rf_metrics']['r2']:.4f}")
                st.metric("Accuracy", f"{st.session_state['rf_metrics']['r2']*100:.2f}%")

            # Comparison Chart
            st.markdown("---")
            comparison_df = pd.DataFrame({
                'Model': ['Gradient Boosted Trees', 'Decision Tree', 'Random Forest'],
                'RMSE': [st.session_state['gbt_metrics']['rmse'], st.session_state['dt_metrics']['rmse'], st.session_state['rf_metrics']['rmse']],
                'MAE': [st.session_state['gbt_metrics']['mae'], st.session_state['dt_metrics']['mae'], st.session_state['rf_metrics']['mae']],
                'R² Score': [st.session_state['gbt_metrics']['r2'], st.session_state['dt_metrics']['r2'], st.session_state['rf_metrics']['r2']],
                'Accuracy (%)': [st.session_state['gbt_metrics']['r2']*100, st.session_state['dt_metrics']['r2']*100, st.session_state['rf_metrics']['r2']*100]
            })

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('RMSE Comparison', 'MAE Comparison', 'R² Score', 'Accuracy (%)'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )

            fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['RMSE'], name='RMSE',
                                marker_color=['blue', 'orange', 'green']), row=1, col=1)
            fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['MAE'], name='MAE',
                                marker_color=['blue', 'orange', 'green']), row=1, col=2)
            fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['R² Score'], name='R²',
                                marker_color=['blue', 'orange', 'green']), row=2, col=1)
            fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['Accuracy (%)'], name='Accuracy',
                                marker_color=['blue', 'orange', 'green']), row=2, col=2)

            fig.update_layout(height=600, showlegend=False, title_text="Model Performance Metrics")
            st.plotly_chart(fig, use_container_width=True)

            # Winner
            import builtins
            best_r2 = builtins.max(st.session_state['gbt_metrics']['r2'], st.session_state['dt_metrics']['r2'], st.session_state['rf_metrics']['r2'])
            if best_r2 == st.session_state['gbt_metrics']['r2']:
                best_model = "Gradient Boosted Trees"
            elif best_r2 == st.session_state['dt_metrics']['r2']:
                best_model = "Decision Tree"
            else:
                best_model = "Random Forest"
            st.success(f"🏆 Best Performing Model: **{best_model}**")

    # PREDICTIONS PAGE
    elif page == "Predictions":
        st.header("Make Predictions")

        if 'rf_model' not in st.session_state or 'appliance_indexer' not in st.session_state:
            st.warning("Please train the models first in the 'Model Training' section.")
            return

        st.subheader("Input Features for Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            appliance = st.selectbox("Appliance Type",
                                    ["Refrigerator", "Air Conditioner", "Washing Machine", "Heater", "Television"])
            temperature = st.slider("Outdoor Temperature (°C)", -10, 45, 20)
            season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Fall"])

        with col2:
            household_size = st.slider("Household Size", 1, 10, 4)
            hour = st.slider("Hour of Day", 0, 23, 12)
            month = st.slider("Month", 1, 12, 6)

        with col3:
            day_of_week = st.slider("Day of Week (1=Monday)", 1, 7, 3)
            model_choice = st.radio("Choose Model", ["Gradient Boosted Trees", "Decision Tree", "Random Forest"])

        if st.button("Predict Energy Consumption", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    # Get spark session
                    spark = init_spark()

                    # Create input DataFrame
                    input_data = spark.createDataFrame([
                        (appliance, temperature, season, household_size, hour, month, day_of_week)
                    ], ["appliance_type", "outdoor_temperature", "season", "household_size", "hour", "month", "day_of_week"])

                    # Apply transformers
                    input_data = st.session_state['appliance_indexer'].transform(input_data)
                    input_data = st.session_state['season_indexer'].transform(input_data)

                    # Create interaction features (same as training)
                    input_data = input_data.withColumn("temp_squared", col("outdoor_temperature") * col("outdoor_temperature"))

                    input_data = st.session_state['assembler'].transform(input_data)
                    input_data = st.session_state['scaler'].transform(input_data)                    # Make prediction
                    if model_choice == "Gradient Boosted Trees":
                        prediction_df = st.session_state['gbt_model'].transform(input_data)
                    elif model_choice == "Decision Tree":
                        prediction_df = st.session_state['dt_model'].transform(input_data)
                    else:
                        prediction_df = st.session_state['rf_model'].transform(input_data)

                    prediction = prediction_df.select("prediction").collect()[0][0]

                    st.markdown("---")
                    st.subheader("Prediction Results")

                    # Get average for comparison from training data
                    training_data = st.session_state.get('training_data', df_filtered)
                    avg_result = training_data.filter(col("appliance_type") == appliance).agg(
                        avg("energy_consumption")
                    ).collect()[0][0]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Consumption", f"{prediction:.2f} kWh")
                    with col2:
                        if avg_result is not None:
                            st.metric("Average for Appliance", f"{avg_result:.2f} kWh")
                        else:
                            st.metric("Average for Appliance", "N/A")
                    with col3:
                        if avg_result is not None:
                            diff = ((prediction - avg_result) / avg_result * 100)
                            st.metric("Difference from Average", f"{diff:+.1f}%")
                        else:
                            st.metric("Difference from Average", "N/A")

                    st.info(f"✨ Prediction made using **{model_choice}** model")

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

    # VISUALIZATIONS PAGE
    elif page == "Visualizations":
        st.header("Advanced Visualizations")

        viz_type = st.selectbox(
            "Select Visualization:",
            ["Correlation Heatmap", "Distribution Analysis", "Time Series", "3D Scatter Plot"]
        )

        if viz_type == "Correlation Heatmap":
            st.subheader("Feature Correlation Heatmap")

            numeric_data = df_filtered.select(
                "energy_consumption", "outdoor_temperature", "household_size", "hour", "month"
            ).toPandas()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)

        elif viz_type == "Distribution Analysis":
            st.subheader("Energy Consumption Distribution")

            sample_data = df_filtered.select("energy_consumption").sample(False, 0.1).toPandas()

            fig = px.histogram(sample_data, x='energy_consumption', nbins=50,
                             title='Distribution of Energy Consumption',
                             labels={'energy_consumption': 'Energy Consumption (kWh)'},
                             color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Time Series":
            st.subheader("Energy Consumption Over Time")

            time_data = df_filtered.groupBy("date", "hour").agg(
                avg("energy_consumption").alias("avg_consumption")
            ).orderBy("date", "hour").limit(500).toPandas()

            fig = px.line(time_data, x='hour', y='avg_consumption', color='date',
                        title='Hourly Energy Consumption Patterns',
                        labels={'avg_consumption': 'Avg Consumption (kWh)', 'hour': 'Hour'})
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "3D Scatter Plot":
            st.subheader("3D Relationship: Temperature, Hour, Consumption")

            sample_3d = df_filtered.select(
                "outdoor_temperature", "hour", "energy_consumption"
            ).sample(False, 0.01).toPandas()

            fig = px.scatter_3d(sample_3d, x='outdoor_temperature', y='hour', z='energy_consumption',
                              color='energy_consumption',
                              title='3D View: Temperature, Hour, and Consumption',
                              labels={'outdoor_temperature': 'Temperature (°C)',
                                     'hour': 'Hour of Day',
                                     'energy_consumption': 'Consumption (kWh)'},
                              color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>Energy Consumption Forecasting System | BDA Lab Mini Project | Powered by PySpark & Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
