# energy_forecast_pyspark.py
"""
End-to-end PySpark pipeline for Energy Consumption Forecasting
Input Parquet columns:
  Home ID | Appliance Type | Energy Consumption (kWh) | Time | Date |
  Outdoor Temperature (°C) | Season | Household Size

Usage:
  spark-submit --master local[*] --driver-memory 8g energy_forecast_pyspark.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, to_timestamp, col, hour, dayofweek, month, when, lit, date_format
from pyspark.sql.functions import unix_timestamp, avg as _avg
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DoubleType

# -------------------------
# 1) Start Spark
# -------------------------
spark.stop()
spark = SparkSession.builder \
    .appName("EnergyConsumptionForecast") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setLogLevel("DEBUG")

# -------------------------
# 2) Read Parquet
# -------------------------
parquet_path = "smart_home_energy_consumption_parquet/"  # change if needed
df = spark.read.parquet(parquet_path)

# quick schema check
print("Input schema:")
df.printSchema()

# -------------------------
# 3) Build timestamp and basic cleaning
# -------------------------
# Assume Date is in ISO or dd/MM/yyyy; adapt format if different.
# We'll concat Date + Time and parse with to_timestamp (flexible if ISO-like).
df = df.withColumn("datetime_str", concat_ws(" ", col("Date"), date_format(col("Time"), "HH:mm:ss")))

# Try to parse — if your Date format is different, adjust format string.
# I'll attempt generic to_timestamp first (Spark will infer common iso formats).
df = df.withColumn("ts", to_timestamp(col("datetime_str"), "yyyy-MM-dd HH:mm:ss"))

# If ts is null for many rows, you may need a custom format like: to_timestamp(col, 'd/M/yyyy H:mm:ss')
null_ts_count = df.filter(col("ts").isNull()).count()
print(f"Rows with NULL parsed timestamp: {null_ts_count}")

# If many nulls, the user should inspect date/time format. For now, continue on parsed rows.
df = df.filter(col("ts").isNotNull())

# Cast numeric columns to double if not already
df = df.withColumn("EnergyConsumption", col("Energy Consumption (kWh)").cast(DoubleType())) \
       .withColumn("OutdoorTemp", col("Outdoor Temperature (°C)").cast(DoubleType())) \
       .withColumn("HouseholdSize", col("Household Size").cast(DoubleType()))

# Rename columns for easier handling
df = df.withColumnRenamed("Home ID", "home_id") \
       .withColumnRenamed("Appliance Type", "appliance") \
       .withColumnRenamed("Season", "season")

# Select relevant columns
df = df.select("home_id", "appliance", "EnergyConsumption", "ts", "OutdoorTemp", "season", "HouseholdSize")

# -------------------------
# 4) Optional: Aggregate to hourly per home (recommended for forecasting)
# -------------------------
# Aggregate (mean) energy consumption to hourly windows per home to reduce noise / handle variable freq
from pyspark.sql.functions import window

hourly = df.groupBy("home_id", window(col("ts"), "60 minutes").alias("w")) \
           .agg(
               _avg("EnergyConsumption").alias("consumption"),
               _avg("OutdoorTemp").alias("outdoor_temp"),
               _avg("HouseholdSize").alias("household_size"),
               # for categorical fields like appliance/season we keep the most frequent value per hour via avg not ideal;
               # instead take first value using min on appliance string hash or just keep NULLABLE. To keep simple:
               # we will ignore appliance here to keep per-home aggregated demand.
           ) \
           .select(
               col("home_id"),
               col("w.start").alias("ts"),
               col("consumption"),
               col("outdoor_temp"),
               col("household_size")
           ) \
           .orderBy("home_id", "ts")

print("After hourly aggregation row count:", hourly.count())

# -------------------------
# 5) Feature engineering: lag features & rolling statistics per home
# -------------------------
from pyspark.sql.functions import lag, avg, stddev
w = Window.partitionBy("home_id").orderBy("ts")

# Create lag features (1 hour, 24 hours)
hourly = hourly.withColumn("lag_1", lag("consumption", 1).over(w)) \
               .withColumn("lag_24", lag("consumption", 24).over(w))

# Rolling mean of past 24 hours (use rowsBetween)
w_range = Window.partitionBy("home_id").orderBy("ts").rowsBetween(-23, 0)  # past 24 including current
hourly = hourly.withColumn("rolling_mean_24", avg("consumption").over(w_range)) \
               .withColumn("rolling_std_24", stddev("consumption").over(w_range))

# Time features
hourly = hourly.withColumn("hour", hour(col("ts"))) \
               .withColumn("dayofweek", dayofweek(col("ts"))) \
               .withColumn("month", month(col("ts"))) \
               .withColumn("is_weekend", when(col("dayofweek").isin(1,7), 1.0).otherwise(0.0))

# Drop rows with nulls caused by lags/rolling windows
hourly_clean = hourly.na.drop(subset=["consumption", "lag_1", "lag_24", "rolling_mean_24"])

print("Rows after dropping nulls (lags/rolling windows):", hourly_clean.count())

# -------------------------
# 6) Train/Val/Test split (time-based)
# -------------------------
# We'll compute global timestamp percentiles and split by timestamp to avoid leakage.
ts_stats = hourly_clean.selectExpr("min(ts) as min_ts", "max(ts) as max_ts").collect()[0]
min_ts = ts_stats["min_ts"]
max_ts = ts_stats["max_ts"]
print("Time range:", min_ts, max_ts)

# compute split datetimes (80 / 10 / 10)
# convert to unix timestamps
min_unix = hourly_clean.agg({"ts":"min"}).collect()[0][0].timestamp()
max_unix = hourly_clean.agg({"ts":"max"}).collect()[0][0].timestamp()
range_unix = max_unix - min_unix
train_end_unix = min_unix + range_unix * 0.8
val_end_unix = min_unix + range_unix * 0.9

# create numeric unix column
hourly_clean = hourly_clean.withColumn("ts_unix", unix_timestamp(col("ts")).cast("long"))

train = hourly_clean.filter(col("ts_unix") <= lit(int(train_end_unix)))
val   = hourly_clean.filter((col("ts_unix") > lit(int(train_end_unix))) & (col("ts_unix") <= lit(int(val_end_unix))))
test  = hourly_clean.filter(col("ts_unix") > lit(int(val_end_unix)))

print("Train / Val / Test sizes:", train.count(), val.count(), test.count())

# -------------------------
# 7) Assemble features
# -------------------------
feature_cols = [
    "lag_1", "lag_24", "rolling_mean_24", "rolling_std_24",
    "outdoor_temp", "household_size", "hour", "dayofweek", "month", "is_weekend"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")

# Optional scaling
scaler = StandardScaler(inputCol="features_vec", outputCol="features", withMean=True, withStd=True)

# Build pipeline to transform datasets
pipeline = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(train)  # fit only on train

train_t = pipeline_model.transform(train).select("features", col("consumption").alias("label"), "ts", "home_id")
val_t   = pipeline_model.transform(val).select("features", col("consumption").alias("label"), "ts", "home_id")
test_t  = pipeline_model.transform(test).select("features", col("consumption").alias("label"), "ts", "home_id")

# -------------------------
# 8) Models: baseline Linear Regression, RandomForest, GBT
# -------------------------
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_mae  = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")

def compute_mape(df, label_col="label", pred_col="prediction"):
    # MAPE: mean(|(y - yhat)/y|)
    from pyspark.sql.functions import abs
    tmp = df.withColumn("abs_pct_err", abs((col(label_col) - col(pred_col)) / when(col(label_col) == 0, lit(1e-9)).otherwise(col(label_col))))
    return float(tmp.agg({"abs_pct_err":"avg"}).collect()[0][0]) * 100.0

models = {}

# 8.1 Linear Regression baseline
print("Training Linear Regression (baseline)...")
lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=50)
lr_model = lr.fit(train_t)
pred_lr = lr_model.transform(val_t)
rmse_lr = evaluator_rmse.evaluate(pred_lr)
mae_lr  = evaluator_mae.evaluate(pred_lr)
mape_lr = compute_mape(pred_lr)
models['LinearRegression'] = (lr_model, rmse_lr, mae_lr, mape_lr)
print("LinearRegression  VAL  RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.2f}%".format(rmse_lr, mae_lr, mape_lr))

# 8.2 Random Forest
print("Training RandomForestRegressor...")
rfr = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=50, maxDepth=10)
rfr_model = rfr.fit(train_t)
pred_rfr = rfr_model.transform(val_t)
rmse_rfr = evaluator_rmse.evaluate(pred_rfr)
mae_rfr  = evaluator_mae.evaluate(pred_rfr)
mape_rfr = compute_mape(pred_rfr)
models['RandomForest'] = (rfr_model, rmse_rfr, mae_rfr, mape_rfr)
print("RandomForest  VAL  RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.2f}%".format(rmse_rfr, mae_rfr, mape_rfr))

# 8.3 Gradient Boosted Trees
print("Training GBTRegressor...")
gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=50, maxDepth=6)
gbt_model = gbt.fit(train_t)
pred_gbt = gbt_model.transform(val_t)
rmse_gbt = evaluator_rmse.evaluate(pred_gbt)
mae_gbt  = evaluator_mae.evaluate(pred_gbt)
mape_gbt = compute_mape(pred_gbt)
models['GBT'] = (gbt_model, rmse_gbt, mae_gbt, mape_gbt)
print("GBT  VAL  RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.2f}%".format(rmse_gbt, mae_gbt, mape_gbt))

# -------------------------
# 9) Choose best model by RMSE and evaluate on test set
# -------------------------
best_name = min(models.keys(), key=lambda k: models[k][1])  # choose lowest val RMSE
best_model, val_rmse, val_mae, val_mape = models[best_name]
print(f"Selected best model: {best_name} (val RMSE={val_rmse:.4f})")

# Evaluate on test set
pred_test = best_model.transform(test_t)
test_rmse = evaluator_rmse.evaluate(pred_test)
test_mae  = evaluator_mae.evaluate(pred_test)
test_mape = compute_mape(pred_test)
print(f"TEST metrics for {best_name}: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")

# -------------------------
# 10) Save model and results
# -------------------------
model_save_path = f"models/{best_name}_model"
# Remove if already exists when re-running locally, or use overwrite logic in your environment
best_model.write().overwrite().save(model_save_path)
print("Saved best model to:", model_save_path)

# Save predictions (test) with timestamps and home_id
pred_out = pred_test.select("home_id", "ts", "label", "prediction")
pred_out.write.mode("overwrite").parquet("predictions/test_predictions_parquet/")

# Save metrics summary
metrics = [
    (best_name, float(val_rmse), float(val_mae), float(val_mape),
     float(test_rmse), float(test_mae), float(test_mape))
]
metrics_df = spark.createDataFrame(metrics, ["model", "val_rmse", "val_mae", "val_mape", "test_rmse", "test_mae", "test_mape"])
metrics_df.coalesce(1).write.mode("overwrite").csv("model_metrics_summary/", header=True)

print("Saved predictions and metrics.")

# -------------------------
# 11) Quick sample output (show some predictions)
# -------------------------
print("Sample predictions (test):")
pred_out.orderBy("ts").show(10, truncate=False)

# -------------------------
# Finish
# -------------------------
spark.stop()
