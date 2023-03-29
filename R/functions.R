# Creating a function to calculate the RMSE out of sample
rmse_is <- function(estimated_model) {
  sqrt(mean(residuals(estimated_model)^2))
}

# Creating a function to calculate the RMSE out of sample
rmse_oos <- function(actuals, preds) {
  sqrt(mean((preds - actuals)^2))
}

# k-Fold Cross-Validation
fold_i_pe <- function(i, k, estimated_model, dataset, outcome) {
  folds <- cut(1:nrow(dataset), breaks=k, labels=FALSE)
  test_indices <- which(folds==i)
  test_set <- dataset[test_indices, ]
  train_set <- dataset[-test_indices, ]
  trained_model <- update(estimated_model, data = train_set)
  predictions <- predict(trained_model, test_set)
  dataset[test_indices, outcome] - predictions # predictive error
}

k_fold_rmse <- function(estimated_model, dataset, outcome, k=10) {
  shuffled_indicies <- sample(1:nrow(dataset))
  dataset <- dataset[shuffled_indicies,]
  
  fold_pred_errors <- sapply(1:k, \(kth) {
    fold_i_pe(kth, k, estimated_model, dataset, outcome)
  })
  
  pred_errors <- unlist(fold_pred_errors)
  rmse <- \(errs) sqrt(mean(errs^2))
  c(rmse_is = rmse(residuals(estimated_model)), rmse_oos = rmse(pred_errors))
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
