# Code-of-research-about-water-yield
Code of "Contrasting Impacts of Climate and Vegetation Changes on Water Yield across Global and Regional Scales"
# Divergent Impacts of Climate and Vegetation Changes on Water Yield

---

## Introduction

This repository contains the codes used in the study:

“Divergent impacts of climate and vegetation changes on water yield across global and regional scales”

The main workflows include:

1. Slope and coefficient of variatio calculation
2. Mann–Kendall test and Sen’s slope analysis
3. Random forest attribution analysis
4. Structural equation modeling

The study aims to quantify the impacts of climate and vegetation changes on water yield.

---

# Data

The repository uses annual raster datasets from 1982–2020.

Main variables include:

| Variable | Description |
|---|---|
| WY | Water yield |
| pdsi | Palmer Drought Severity Index |
| srad | Surface shortwave radiation |
| soil | Soil moisture |
| tmean | Mean air temperature |
| vpd | Vapor pressure deficit |
| ws | Wind speed |
| LAI | Leaf area index |

Processed datasets are available from the Mendeley Data repository:

DOI: 10.17632/48wyypr7sd.1

---

# Scripts

## 1. Slope+CV.py

This script calculates:

- spatial mean values
- linear trend slopes
- coefficient of variation (CV)

for each variable during 1982–2020.

Output:

results/slope_cv_summary.csv

---

## 2. Sen+M-K.py

This script performs:

- Mann–Kendall significance testing
- Sen’s slope estimation

for raster time series.

Outputs include:

- slope raster
- p-value raster
- z-statistic raster
- significance mask raster

Output directory:

results/trend_analysis/

---

## 3. RF.py

This script trains and applies the random forest model for climate-driven water yield prediction.

Method:

Pixels with minimal vegetation change (LAI slope between −0.01 and 0.01) are selected as climate-dominated regions.

The random forest model is trained using climate factors and then applied to the entire study area to estimate climate-driven water yield trend.

climate-driven water yield trend = Observed yield trend − Predicted climate-driven yield trend

Outputs:

models/random_forest_model.pkl
results/predicted_climate_driven_yield_trend.tif
results/feature_importance.csv
results/model_metrics.csv

---

## 4. SEM.R

This script performs structural equation modeling (SEM) to evaluate the direct and indirect effects of climate and vegetation factors on water yield.

The workflow includes:

- VIF analysis
- SEM fitting
- path coefficient extraction
- model fit evaluation
- SEM path plotting

Outputs:

results/sem/

---

# Note

Before running the scripts:

1. Ensure all raster datasets have identical:
   - projection
   - spatial resolution
   - spatial extent

2. Organize the datasets according to the repository structure.

3. Update local paths if necessary.

---
