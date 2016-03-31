
#' @title Model Info
#' @description Provide a list of available models and respective information
#' @return A data.frame containing:
#' \item{methods}{The abbreviated code for the method}
#' \item{description}{Full name of the method}
#' \item{hyperparams}{Comma separated string of hyperparameters for tuning}
#' @author Dr. Charles Determan Jr. PhD
#' @export
modelInfo <- function(){
    methods <- c(
        "neuralnet",
        "rf"
        )
    description <- c(
        "Neural Network",
        "Random Forest"
        )
    hyperparams <- c(
        ".hidden,.threshold",
        ".mtry"
        )
    
    return(data.frame(methods, description, hyperparams))
}