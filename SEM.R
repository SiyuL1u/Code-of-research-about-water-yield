# =========================================================
# Structural Equation Modeling (SEM)
# =========================================================
# This script performs SEM analysis to evaluate
# the direct and indirect effects of climate
# and vegetation changes on water yield trends.
#
# Author: Your Name
# R version: 4.x
# =========================================================

# =========================================================
# Load Required Packages
# =========================================================

library(lavaan)
library(semPlot)
library(ggplot2)
library(dplyr)
library(readr)
library(car)

# =========================================================
# Set Paths
# =========================================================

data_path <- file.path(
  "data",
  "sem",
  "SEM_dataset_standardized.csv"
)

result_dir <- file.path(
  "results",
  "sem"
)

dir.create(result_dir,
           recursive = TRUE,
           showWarnings = FALSE)

# =========================================================
# Read Dataset
# =========================================================

dataset <- read.csv(
  data_path,
  header = TRUE,
  stringsAsFactors = FALSE
)

dataset <- as.data.frame(dataset)

dataset_clean <- na.omit(dataset)

cat("\nDataset loaded successfully.\n")

cat("\nNumber of valid samples:",
    nrow(dataset_clean), "\n")

# =========================================================
# Check Dataset Structure
# =========================================================

print(head(dataset_clean))

print(colnames(dataset_clean))

str(dataset_clean)

# =========================================================
# Multicollinearity Check
# =========================================================

vif_model <- lm(
  WY_slope ~
    pdsi_slope +
    srad_slope +
    soil_slope +
    tmean_slope +
    vpd_slope +
    ws_slope +
    LAI_slope,
  data = dataset_clean
)

vif_values <- vif(vif_model)

print(vif_values)

# Save VIF results
write.csv(
  data.frame(
    Variable = names(vif_values),
    VIF = vif_values
  ),
  file.path(result_dir, "vif_results.csv"),
  row.names = FALSE
)

# =========================================================
# Define SEM Model
# =========================================================

sem_model <- '

  # Direct effects on water yield
  WY_slope ~
    pdsi_slope +
    srad_slope +
    tmean_slope +
    soil_slope +
    vpd_slope +
    ws_slope +
    LAI_slope

  # Climate effects on vegetation
  LAI_slope ~
    pdsi_slope +
    srad_slope +
    soil_slope +
    ws_slope

  # Climate effects on soil moisture
  soil_slope ~
    pdsi_slope +
    tmean_slope +
    vpd_slope

'

# =========================================================
# Fit SEM Model
# =========================================================

fit_sem <- sem(
  sem_model,
  data = dataset_clean
)

# =========================================================
# Model Summary
# =========================================================

summary_output <- capture.output(
  summary(
    fit_sem,
    fit.measures = TRUE,
    standardized = TRUE
  )
)

writeLines(
  summary_output,
  file.path(result_dir, "sem_summary.txt")
)

cat("\nSEM model fitted successfully.\n")

# =========================================================
# Extract Standardized Path Coefficients
# =========================================================

std_solution <- standardizedSolution(
  fit_sem
)

write.csv(
  std_solution,
  file.path(
    result_dir,
    "standardized_path_coefficients.csv"
  ),
  row.names = FALSE
)

# =========================================================
# Extract Fit Indices
# =========================================================

fit_indices <- fitMeasures(
  fit_sem,
  c(
    "chisq",
    "df",
    "pvalue",
    "cfi",
    "tli",
    "rmsea",
    "srmr"
  )
)

write.csv(
  data.frame(
    Index = names(fit_indices),
    Value = fit_indices
  ),
  file.path(
    result_dir,
    "fit_indices.csv"
  ),
  row.names = FALSE
)

# =========================================================
# Generate SEM Path Diagram
# =========================================================

png(
  filename = file.path(
    result_dir,
    "sem_path_diagram.png"
  ),
  width = 3000,
  height = 2200,
  res = 300
)

semPaths(
  fit_sem,
  what = "std",
  layout = "tree",
  edge.label.cex = 1.2,
  fade = FALSE,
  residuals = FALSE,
  exoCov = FALSE
)

dev.off()

cat("\nSEM path diagram saved.\n")

# =========================================================
# Finished
# =========================================================

cat("\nSEM workflow completed successfully.\n")
