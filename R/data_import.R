### Packages and Data

# Dataset available at: https://github.com/danielemelotti/decision_tree_exploration

## Installing the necessary packages:

#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("randomForest")
#install.packages("ranger")
#install.packages("insight")
#install.packages("formula.tools")
#install.packages("caret")
require(rpart)
require(rpart.plot)
require(dplyr)
require(randomForest)
require(ranger)
require(insight)
require(formula.tools)
require(caret)

# For the purpose of this exercise, we'll be using a dataset regarding housing prices in Miami.
# The dataset includes a total of 17 variables.

## Loading specific data and model_formula
download_data <- read.csv("data/miami-housing.csv")
str(download_data)

# Filtering for the variables of interest
fulldata_ordered <- download_data %>%
  select(-"PARCELNO")

# Shuffling the data to increase randomness
fulldata <- fulldata_ordered[sample(seq_len(nrow(fulldata_ordered))), ]

str(fulldata)

head(fulldata)

dv <- "SALE_PRC" # outcome variable
model_formula <- SALE_PRC ~ LATITUDE + LONGITUDE + LND_SQFOOT + TOT_LVG_AREA + SPEC_FEAT_VAL + RAIL_DIST + OCEAN_DIST + WATER_DIST + CNTR_DIST + SUBCNTR_DI + HWY_DIST + age + avno60plus + month_sold + structure_quality

xv_list <- find_predictors(model_formula) # list of predictors
# xv_form <- formula(model_formula)[-2] # predictors as formula (but with tilde)
# xv_call <- formula.tools::rhs(model_formula) # predictors call

table(is.na(fulldata)) # no missing values
