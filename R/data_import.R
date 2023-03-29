### Packages and Data

# Dataset available at: https://github.com/danielemelotti/decision_tree_exploration

## Installing the necessary packages:

#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("rattle")
#install.packages("randomForest")
#install.packages("ranger")
require(rpart)
require(rpart.plot)
require(rattle)
require(dplyr)
require(randomForest)
require(ranger)

# For the purpose of this exercise, we'll be using a dataset regarding housing prices in Miami.
# The dataset includes a total of 17 variables.

## Loading specific data and model_formula
download_data <- read.csv("miami-housing.csv")
str(download_data)

# Filtering for the variables of interest
fulldata <- download_data %>%
  select(-"PARCELNO")

str(fulldata)

dv <- "SALE_PRC" # outcome variable
model_formula <- SALE_PRC ~ LND_SQFOOT + TOT_LVG_AREA + SPEC_FEAT_VAL + RAIL_DIST + OCEAN_DIST +
  WATER_DIST + CNTR_DIST + SUBCNTR_DI + HWY_DIST + age + avno60plus + structure_quality

table(is.na(fulldata)) # no missing values

# Performing a 75:25 split
set.seed(2012)
train_indices <- sample(1:nrow(fulldata), size = 0.75 * nrow(fulldata))

# Creating the train and test sets
train_set <- fulldata[train_indices, ]
test_set <- fulldata[-train_indices, ]
