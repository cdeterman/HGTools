#' @title MAD - Mean Absolute Deviation
#' @description Returns the average difference between a model's prediction and
#' the actual testing data.
#' @param model A fitted model (e.g. lm, glm, etc.)
#' @param raw_data The results that you wish to compare the predicted results
#' @param expo Whether the results of the prediction should be raised to the exponent.
#' Such as when you fit a model with the log of the dependent variable. Default \code{"expo = TRUE"}.
#' @export

mad_model <- function(model, raw_data, expo = TRUE){
  diff <- abs(raw_data - exp(predict(model)))
  return(mean(diff))
}

#' @title Delta Models
#' @description Returns a dataframe of delta R2 and p-value of an added Independent Variable
#' to a base model.
#' @param model A fitted lm model to be used as a base comparison
#' @param new_iv A string name of the new variable to be added
#' @export

delta_models <- function(base_model, new_iv){
  if(class(base_model) != "lm"){
    simpleError("The model provided is not 'lm'.  Please fit model with the lm function first.")
  }
  old_r2 <- summary(base_model)$adj.r.squared
  iv <- as.name(new_iv)
  # update model to include new_iv
  new_mod <- update(base_model, bquote(.~. + .(iv)))
  # coefficients matrix
  coef_df <- as.data.frame(summary(new_mod)$coefficients)
  # extract p-value
  new_pv <- coef_df[rownames(coef_df) == iv, 4]
  # extract adj.r.squared
  new_r2 <- summary(new_mod)$adj.r.squared
  # initialize blank df
  output <- data.frame(iv="blank", delta_r2=0, p_value=0)
  output$iv <- new_iv
  output$delta_r2 <- new_r2-old_r2
  output$p_value <- new_pv
  return(output)
}