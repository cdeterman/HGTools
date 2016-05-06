#' @title List Current Model Types
#' @description The function provides a list of all currently implemented models
#' and a brief description of each.
#' @return A data.frame containing the following columns:
#' \item{model_types}{Code for each model type}
#' \item{description}{Brief detail for each model}
#' @export
model_types <- function(){
    models <- c("payor", "binary")
    labels <- c("Predicting Payor Types (Commercial, Medicare, Medicaid, & Self)",
                "Generic Binary Model")
    out <- data.frame(model_types = models, description = labels)
    return(out)
}
