import os
import glob
import numpy as np
import rasterio
import pandas as pd
from scipy.stats import linregress

# ======================
# Configuration
# ======================

BASE_FOLDER = os.path.join("data", "factor")

FACTORS = {
    "water_yield": "WY",
    "pdsi": "pdsi",
    "shortwave_radiation": "srad",
    "soil_moisture": "soil",
    "mean_temperature": "tmean",
    "vapor_pressure_deficit": "vpd",
    "wind_speed": "ws",
    "lai": "LAI"
}

YEARS = list(range(1982, 2021))

# ======================
# Functions
# ======================

def calculate_yearly_mean(folder_path):
    """
    Calculate yearly spatial mean values from raster files.
    """
    file_list = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

    yearly_means = []
    available_years = []

    for file in file_list:
        filename = os.path.basename(file)

        try:
            year = int(filename.split('_')[-1].split('.')[0])

            if year in YEARS:
                with rasterio.open(file) as src:
                    data = src.read(1).astype(float)
                    data[data == src.nodata] = np.nan

                yearly_means.append(np.nanmean(data))
                available_years.append(year)

        except ValueError:
            print(f"Skipping file: {file}")

    return available_years, yearly_means


# ======================
# Main Calculation
# ======================

results = []

for factor_name, folder_name in FACTORS.items():

    factor_path = os.path.join(BASE_FOLDER, folder_name)

    years, values = calculate_yearly_mean(factor_path)

    values = np.array(values)

    # Linear trend
    if len(values) > 1:
        slope, _, _, _, _ = linregress(years, values)
    else:
        slope = np.nan

    # Coefficient of variation
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)

    cv = std_val / mean_val if mean_val != 0 else np.nan

    results.append({
        "Factor": factor_name,
        "Mean": mean_val,
        "Std": std_val,
        "CV": cv,
        "Trend_Slope": slope
    })

# ======================
# Save Results
# ======================

result_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

output_path = os.path.join("results", "trend_cv_summary.csv")

result_df.to_csv(output_path, index=False)

print(result_df)

print(f"\nResults saved to: {output_path}")
