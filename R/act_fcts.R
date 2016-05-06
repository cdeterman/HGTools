#' @title List Current Activation Functions
#' @description The function provides a list of all currently implemented
#' activation functions and a brief description.
#' @return A data.frame containing the following columns:
#' \item{act_fct}{Code for each activation function}
#' \item{description}{Brief detail for each model}
#' @export
act_fcts <- function(){
    acts <- c("tanh", "logistic", "relu")
    labels <- c("Hyperbolic Tangent", "Logistic", "Rectified Linear Units")
    out <- data.frame(act_fcts = acts, description = labels)
    out$act_fcts <- as.character(out$act_fcts)
    return(out)
}