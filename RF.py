
# =========================================================
# Import Packages
# =========================================================

import os
import joblib
import rasterio
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================================================
# Configuration
# =========================================================

# Project directories
TRAINING_DIR = os.path.join("data", "training")
PREDICTION_DIR = os.path.join("data", "prediction")

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Random forest parameters
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# Predictor variables
FEATURES = [
    "pdsi_mean",
    "pdsi_slope",
    "srad_mean",
    "srad_slope",
    "soil_mean",
    "soil_slope",
    "tmean_mean",
    "tmean_slope",
    "vpd_mean",
    "vpd_slope",
    "ws_mean",
    "ws_slope"
]

TARGET = "WY_slope"

# =========================================================
# Raster File Definitions
# =========================================================

TRAINING_RASTERS = {
    "pdsi_mean": "cli_pdsi_mean.tif",
    "pdsi_slope": "cli_pdsi_slope.tif",
    "srad_mean": "cli_srad_mean.tif",
    "srad_slope": "cli_srad_slope.tif",
    "soil_mean": "cli_soil_mean.tif",
    "soil_slope": "cli_soil_slope.tif",
    "tmean_mean": "cli_tmean_mean.tif",
    "tmean_slope": "cli_tmean_slope.tif",
    "vpd_mean": "cli_vpd_mean.tif",
    "vpd_slope": "cli_vpd_slope.tif",
    "ws_mean": "cli_ws_mean.tif",
    "ws_slope": "cli_ws_slope.tif",
    "yield_slope": "cli_yield_slope.tif"
}

PREDICTION_RASTERS = {
    "pdsi_mean": "pdsi_mean.tif",
    "pdsi_slope": "pdsi_slope.tif",
    "srad_mean": "srad_mean.tif",
    "srad_slope": "srad_slope.tif",
    "soil_mean": "soil_mean.tif",
    "soil_slope": "soil_slope.tif",
    "tmean_mean": "tmean_mean.tif",
    "tmean_slope": "tmean_slope.tif",
    "vpd_mean": "vpd_mean.tif",
    "vpd_slope": "vpd_slope.tif",
    "ws_mean": "ws_mean.tif",
    "ws_slope": "ws_slope.tif"
}

# =========================================================
# Functions
# =========================================================

def load_rasters(data_dir, raster_dict):
    """
    Load raster files and convert to DataFrame.
    """

    data = {}
    meta = None

    for variable, filename in raster_dict.items():

        raster_path = os.path.join(data_dir, filename)

        with rasterio.open(raster_path) as src:

            array = src.read(1).astype(np.float32)

            array[array == src.nodata] = np.nan

            data[variable] = array.flatten()

            if meta is None:
                meta = src.meta.copy()

    df = pd.DataFrame(data)

    return df, meta


def save_raster(output_path, array, meta):
    """
    Save array as GeoTIFF.
    """

    meta.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=np.nan
    )

    with rasterio.open(output_path, "w", **meta) as dst:

        dst.write(
            array.astype(np.float32),
            1
        )


# =========================================================
# Step 1: Load Training Data
# =========================================================

print("\nLoading training rasters...")

train_df, raster_meta = load_rasters(
    TRAINING_DIR,
    TRAINING_RASTERS
)

# Remove invalid pixels
train_df = train_df.dropna().reset_index(drop=True)

print(f"Valid training samples: {len(train_df)}")

# =========================================================
# Step 2: Prepare Training Dataset
# =========================================================

X = train_df[FEATURES]

y = train_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# =========================================================
# Step 3: Train Random Forest
# =========================================================

print("\nTraining Random Forest model...")

rf_model = RandomForestRegressor(**RF_PARAMS)

rf_model.fit(X_train, y_train)

print("Model training completed.")

# =========================================================
# Step 4: Model Evaluation
# =========================================================

print("\nEvaluating model...")

y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

# Save metrics
metrics_df = pd.DataFrame({
    "Metric": ["R2", "MAE", "RMSE"],
    "Value": [r2, mae, rmse]
})

metrics_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "model_metrics.csv"
    ),
    index=False
)

# =========================================================
# Step 5: Feature Importance
# =========================================================

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "feature_importance.csv"
    ),
    index=False
)

print("\nFeature importance saved.")

# =========================================================
# Step 6: Save Model
# =========================================================

model_path = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)

joblib.dump(rf_model, model_path)

print(f"\nModel saved to:\n{model_path}")

# =========================================================
# Step 7: Load Prediction Data
# =========================================================

print("\nLoading prediction rasters...")

pred_df, pred_meta = load_rasters(
    PREDICTION_DIR,
    PREDICTION_RASTERS
)

# Record valid pixels
valid_mask = ~pred_df.isnull().any(axis=1)

pred_valid = pred_df.loc[
    valid_mask,
    FEATURES
]

print(f"Valid prediction pixels: {len(pred_valid)}")

# =========================================================
# Step 8: Predict Climate-Driven Yield Trend
# =========================================================

print("\nPredicting climate-driven water yield trends...")

prediction_array = np.full(
    pred_df.shape[0],
    np.nan
)

prediction_array[valid_mask] = rf_model.predict(
    pred_valid
)

prediction_array = prediction_array.reshape(
    pred_meta["height"],
    pred_meta["width"]
)

# =========================================================
# Step 9: Save Prediction Raster
# =========================================================

prediction_output = os.path.join(
    RESULT_DIR,
    "predicted_climate_driven_yield_trend.tif"
)

save_raster(
    prediction_output,
    prediction_array,
    pred_meta
)

print("\nPrediction raster saved.")

# =========================================================
# Finished
# =========================================================

print("\nWorkflow completed successfully.")
