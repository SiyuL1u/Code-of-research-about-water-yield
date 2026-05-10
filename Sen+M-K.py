import os
import numpy as np
import rasterio
import pymannkendall as mk

# =========================================================
# Configuration
# =========================================================

# Input and output directories
BASE_INPUT_DIR = "data"
OUTPUT_DIRECTORY = os.path.join(
    "results",
    "trend_analysis"
)

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Variables to process
VARIABLES = [
    "yield",
    "pdsi",
    "srad",
    "soil",
    "tmean",
    "vpd",
    "ws",
    "LAI"
]

# Analysis period
YEARS = list(range(1982, 2021))

# =========================================================
# Functions
# =========================================================

def read_rasters(input_dir, variable_name, years):
    """
    Read annual raster data into a 3D array.
    """

    rasters = []
    profile = None

    for year in years:

        raster_path = os.path.join(
            input_dir,
            variable_name,
            f"{variable_name}_{year}.tif"
        )

        with rasterio.open(raster_path) as src:

            if profile is None:
                profile = src.profile

            data = src.read(1).astype(float)

            data[data == src.nodata] = np.nan

            rasters.append(data)

    return np.array(rasters), profile


def mk_sen_analysis(data):
    """
    Perform Mann-Kendall trend test
    and Sen's slope estimation.
    """

    rows, cols = data.shape[1:]

    z_vals = np.full((rows, cols), np.nan)
    p_vals = np.full((rows, cols), np.nan)
    slopes = np.full((rows, cols), np.nan)

    for i in range(rows):
        for j in range(cols):

            time_series = data[:, i, j]

            time_series = time_series[
                ~np.isnan(time_series)
            ]

            if len(time_series) < 2:
                continue

            result = mk.original_test(time_series)

            z_vals[i, j] = result.z
            p_vals[i, j] = result.p
            slopes[i, j] = result.slope

    return z_vals, p_vals, slopes


def save_raster(output_path, data, profile):
    """
    Save raster output.
    """

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=np.nan
    )

    with rasterio.open(
        output_path,
        'w',
        **profile
    ) as dst:

        dst.write(
            data.astype(np.float32),
            1
        )


# =========================================================
# Main Workflow
# =========================================================

for variable in VARIABLES:

    print(f"\nProcessing variable: {variable}")

    # Read raster stack
    data_stack, raster_profile = read_rasters(
        BASE_INPUT_DIR,
        variable,
        YEARS
    )

    print("Running Mann-Kendall test and Sen's slope analysis...")

    z_raster, p_raster, slope_raster = mk_sen_analysis(
        data_stack
    )

    # Significance mask
    sig_raster = np.where(
        p_raster < 0.05,
        1,
        0
    )

    print("Saving outputs...")

    # Save z-statistic
    save_raster(
        os.path.join(
            OUTPUT_DIRECTORY,
            f"{variable}_z.tif"
        ),
        z_raster,
        raster_profile
    )

    # Save p-value
    save_raster(
        os.path.join(
            OUTPUT_DIRECTORY,
            f"{variable}_p.tif"
        ),
        p_raster,
        raster_profile
    )

    # Save Sen's slope
    save_raster(
        os.path.join(
            OUTPUT_DIRECTORY,
            f"{variable}_slope.tif"
        ),
        slope_raster,
        raster_profile
    )

    # Save significance mask
    save_raster(
        os.path.join(
            OUTPUT_DIRECTORY,
            f"{variable}_significance.tif"
        ),
        sig_raster,
        raster_profile
    )

    print(f"{variable} completed.")

print("\nAll variables processed successfully.")
