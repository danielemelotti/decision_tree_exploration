# Creating a function to calculate the RMSE out of sample
rmse_is <- function(estimated_model) {
  sqrt(mean(residuals(estimated_model)^2))
}

# Creating a function to calculate the RMSE out of sample
rmse_oos <- function(actuals, preds) {
  sqrt(mean((preds - actuals)^2))
}

# Creating the functions for learning and predicting (Bagging)
bagged_learn <- function(estimated_model, dataset, b=100) {
  lapply(1:b, \(i) {
    data_i <- dataset[sample(nrow(dataset), replace=TRUE),]
    update(estimated_model, data=data_i) 
  })
}

bagged_predict <- function(bagged_models, new_data) {
  predictions <- lapply(bagged_models, \(m) predict(m, new_data))
  as.data.frame(predictions) |> apply(FUN=mean, MARGIN=1)
}

# Creating the functions for learning and predicting (Boosting)
boost_learn <- function(estimated_model, dataset, outcome, n=100, rate=0.1) { 
  predictors <- dataset[,-which(names(dataset) == outcome)]
  
  res <- dataset[,outcome]
  models <- list()
  
  for (i in 1:n) {
    new_data <- cbind(res, predictors)
    colnames(new_data)[1] = outcome
    this_model <- update(estimated_model, data = new_data)
    res <- res - rate * predict(this_model, dataset)
    models[[i]] <- this_model
  }
  
  list(models=models, rate=rate)
}

boost_predict <- function(boosted_learning, new_data) {
  boosted_models <- boosted_learning$models
  rate <- boosted_learning$rate
  predictions <- lapply(boosted_models, \(this_model) {
    predict(this_model, new_data)
  })
  pred_frame <- as.data.frame(predictions) |> unname()
  apply(pred_frame, FUN = \(preds) rate * sum(preds), MARGIN=1)
}