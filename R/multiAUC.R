
#' @title Multi-class AUC
#' @description This function is very similar to the normal \code{\link[pROC]{auc}} 
#' function in the \code{pROC} package.  It performs multiclass AUC as defined by 
#' Hand and Till (2001) whereby the AUC is calculated for each class and subsequently 
#' average.  This function extends this function to accept a probability matrix.
#' @param classes A vector of 'observed' classes 
#' @param probs A matrix containing a column for each class with respective probabilities
#' @references David J. Hand and Robert J. Till (2001). A Simple Generalisation of the 
#' Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning 
#' 45(2), p. 171–186. DOI: 10.1023/A:1010920819831.
#' @import pROC
#' @import assertive
multiAUC <- function(classes, probs){
    
    assert_is_matrix(probs)
    
    cnames <- colnames(probs)
    aucs <- vector(mode = "numeric", length = ncol(probs))
    
    # print(cnames)
    
    for(i in 1:ncol(probs)){
        obs <- ifelse(classes == cnames[i], 1, 0)
        # print(table(obs))
        aucs[i] <- auc(obs, as.numeric(probs[,cnames[i]]))[1]
    }
    
    return(mean(aucs))
}
