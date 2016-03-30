
# Functions for generating model performance metrics

predictionStats <- function(obs, pred){
    
    print("calling predictionStats")
    
    # F1-score = 2 * (PPV * Sensitivity)/(PPV + Sensitivity)
    # AUC, Sensitivity, Specificity, PPV, NPV, F1-Score
    
    tmp.auc <- colMeans(colAUC(pred, obs, 
                               plotROC = FALSE, alg = "ROC"))
    
#     print("calculated AUC")
    
#     tmp.auc <- colMeans(colAUC(order(pred), obs, 
#                                plotROC = FALSE, alg = "ROC"))
#     confMat <- confusionMatrix(obs, pred)

#     print("observations")
#     print(table(obs))
#     print(length(obs))
#     print("rounded pred")
#     print(table(factor(round(pred), levels = c(0, 1))))
#     print(length(pred))

    confMat <- confusionMatrix(obs, factor(round(pred), levels = c(0, 1)))

#     print("called confusionMatrix")
    
    sensitivity <- confMat$byClass["Sensitivity"]
    specificity <- confMat$byClass["Specificity"]
    ppv <- confMat$byClass["Pos Pred Value"]
    npv <- confMat$byClass["Neg Pred Value"]
    fsc <- 2 * (ppv * sensitivity)/(ppv + sensitivity)
    
    out <- c(tmp.auc, sensitivity, specificity, ppv, npv, fsc)
    names(out) <- c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "F1-Score")
    return(out)
}

confusionMatrix <- function (data, ...) 
{
    UseMethod("confusionMatrix")
}

confusionMatrix.default <- function(
    data, reference, positive = NULL, 
    dnn = c("Prediction", 
            "Reference"), prevalence = NULL, ...) 
{
    if (!is.factor(data)) 
        data <- factor(data)
    if (!is.factor(reference)) 
        reference <- factor(reference)
    if (!is.character(positive) & !is.null(positive)) 
        stop("positive argument must be character")
    if (length(levels(data)) > length(levels(reference))) 
        stop("the data cannot have more levels than the reference")
    if (!any(levels(data) %in% levels(reference))) {
        stop("The data must contain some levels that overlap the reference.")
    }
    if (!all(levels(data) %in% levels(reference))) {
        badLevel <- levels(data)[!levels(data) %in% levels(reference)]
        if (sum(table(data)[badLevel]) > 0) {
            stop("The data contain levels not found in the data.")
        }
        else {
            warning("The data contains levels not found in the data, but they are empty and will be dropped.")
            data <- factor(as.character(data))
        }
    }
    if (any(levels(reference) != levels(data))) {
        warning("Levels are not in the same order for reference and data. Refactoring data to match.")
        data <- as.character(data)
        data <- factor(data, levels = levels(reference))
    }
    classLevels <- levels(data)
    numLevels <- length(classLevels)
    if (numLevels < 2) 
        stop("there must be at least 2 factors levels in the data")
    if (numLevels == 2 & is.null(positive)) 
        positive <- levels(reference)[1]
    classTable <- table(data, reference, dnn = dnn, ...)
    confusionMatrix.table(classTable, positive, prevalence = prevalence)
}

confusionMatrix.table <- function (data, positive = NULL, prevalence = NULL, ...) 
{
    requireNamespace("e1071", quietly = TRUE)
    if (length(dim(data)) != 2) 
        stop("the table must have two dimensions")
    if (!all.equal(nrow(data), ncol(data))) 
        stop("the table must nrow = ncol")
    if (!all.equal(rownames(data), colnames(data))) 
        stop("the table must the same classes in the same order")
    if (!is.character(positive) & !is.null(positive)) 
        stop("positive argument must be character")
    classLevels <- rownames(data)
    numLevels <- length(classLevels)
    if (numLevels < 2) 
        stop("there must be at least 2 factors levels in the data")
    if (numLevels == 2 & is.null(positive)) 
        positive <- rownames(data)[1]
    if (numLevels == 2 & !is.null(prevalence) && length(prevalence) != 
            1) 
        stop("with two levels, one prevalence probability must be specified")
    if (numLevels > 2 & !is.null(prevalence) && length(prevalence) != 
            numLevels) 
        stop("the number of prevalence probability must be the same as the number of levels")
    if (numLevels > 2 & !is.null(prevalence) && is.null(names(prevalence))) 
        stop("with >2 classes, the prevalence vector must have names")
    propCI <- function(x) {
        binom.test(sum(diag(x)), sum(x))$conf.int
    }
    propTest <- function(x) {
        out <- binom.test(sum(diag(x)), sum(x), p = max(apply(x, 
                                                              2, sum)/sum(x)), alternative = "greater")
        unlist(out[c("null.value", "p.value")])
    }
    overall <- c(unlist(e1071::classAgreement(data))[c("diag", 
                                                       "kappa")], propCI(data), propTest(data), mcnemar.test(data)$p.value)
    names(overall) <- c("Accuracy", "Kappa", "AccuracyLower", 
                        "AccuracyUpper", "AccuracyNull", "AccuracyPValue", "McnemarPValue")
    if (numLevels == 2) {
        if (is.null(prevalence)) 
            prevalence <- sum(data[, positive])/sum(data)
        negative <- classLevels[!(classLevels %in% positive)]
        tableStats <- c(sensitivity.table(data, positive), 
                        specificity.table(data, negative), 
                        posPredValue.table(data, positive, prevalence = prevalence), 
                        negPredValue.table(data, negative, prevalence = prevalence), 
                        prevalence, sum(data[positive, positive])/sum(data), 
                        sum(data[positive, ])/sum(data))
        names(tableStats) <- c("Sensitivity", "Specificity", 
                               "Pos Pred Value", "Neg Pred Value", "Prevalence", 
                               "Detection Rate", "Detection Prevalence")
        tableStats["Balanced Accuracy"] <- (tableStats["Sensitivity"] + 
                                                tableStats["Specificity"])/2
    }
    else {
        tableStats <- matrix(NA, nrow = length(classLevels), 
                             ncol = 8)
        for (i in seq(along = classLevels)) {
            pos <- classLevels[i]
            neg <- classLevels[!(classLevels %in% classLevels[i])]
            prev <- if (is.null(prevalence)) 
                sum(data[, pos])/sum(data)
            else prevalence[pos]
            tableStats[i, ] <- c(sensitivity.table(data, pos), 
                                 specificity.table(data, neg), 
                                 posPredValue.table(data, pos, prevalence = prev), 
                                 negPredValue.table(data, 
                                                    neg, prevalence = prev), prev, sum(data[pos, 
                                                                                            pos])/sum(data), sum(data[pos, ])/sum(data), 
                                 NA)
            tableStats[i, 8] <- (tableStats[i, 1] + tableStats[i, 
                                                               2])/2
        }
        rownames(tableStats) <- paste("Class:", classLevels)
        colnames(tableStats) <- c("Sensitivity", "Specificity", 
                                  "Pos Pred Value", "Neg Pred Value", "Prevalence", 
                                  "Detection Rate", "Detection Prevalence", "Balanced Accuracy")
    }
    list(positive = positive, table = data, overall = overall, 
         byClass = tableStats, dots = list(...))
}

sensitivity <- function (data, ...) 
{
    UseMethod("sensitivity")
}

specificity <- function (data, ...) 
{
    UseMethod("specificity")
}

posPredValue <- function (data, ...) 
{
    UseMethod("posPredValue")
}

negPredValue <- function (data, ...) 
{
    UseMethod("negPredValue")
}

sensitivity.table <- function (data, positive = rownames(data)[1], ...) 
{
    if (!all.equal(nrow(data), ncol(data))) 
        stop("the table must have nrow = ncol")
    if (!all.equal(rownames(data), colnames(data))) 
        stop("the table must the same groups in the same order")
    if (nrow(data) > 2) {
        tmp <- data
        data <- matrix(NA, 2, 2)
        colnames(data) <- rownames(data) <- c("pos", "neg")
        posCol <- which(colnames(tmp) %in% positive)
        negCol <- which(!(colnames(tmp) %in% positive))
        data[1, 1] <- sum(tmp[posCol, posCol])
        data[1, 2] <- sum(tmp[posCol, negCol])
        data[2, 1] <- sum(tmp[negCol, posCol])
        data[2, 2] <- sum(tmp[negCol, negCol])
        data <- as.table(data)
        positive <- "pos"
        rm(tmp)
    }
    numer <- sum(data[positive, positive])
    denom <- sum(data[, positive])
    sens <- ifelse(denom > 0, numer/denom, NA)
    sens
}

specificity.table <- function (data, negative = rownames(data)[-1], ...) 
{
    if (!all.equal(nrow(data), ncol(data))) 
        stop("the table must have nrow = ncol")
    if (!all.equal(rownames(data), colnames(data))) 
        stop("the table must the same groups in the same order")
    if (nrow(data) > 2) {
        tmp <- data
        data <- matrix(NA, 2, 2)
        colnames(data) <- rownames(data) <- c("pos", "neg")
        negCol <- which(colnames(tmp) %in% negative)
        posCol <- which(!(colnames(tmp) %in% negative))
        data[1, 1] <- sum(tmp[posCol, posCol])
        data[1, 2] <- sum(tmp[posCol, negCol])
        data[2, 1] <- sum(tmp[negCol, posCol])
        data[2, 2] <- sum(tmp[negCol, negCol])
        data <- as.table(data)
        negative <- "neg"
        rm(tmp)
    }
    numer <- sum(data[negative, negative])
    denom <- sum(data[, negative])
    spec <- ifelse(denom > 0, numer/denom, NA)
    spec
}

posPredValue.table <- function (data, positive = rownames(data)[1], prevalence = NULL, 
                                ...) 
{
    if (!all.equal(nrow(data), ncol(data))) 
        stop("the table must have nrow = ncol")
    if (!all.equal(rownames(data), colnames(data))) 
        stop("the table must the same groups in the same order")
    if (nrow(data) > 2) {
        tmp <- data
        data <- matrix(NA, 2, 2)
        colnames(data) <- rownames(data) <- c("pos", "neg")
        posCol <- which(colnames(tmp) %in% positive)
        negCol <- which(!(colnames(tmp) %in% positive))
        data[1, 1] <- sum(tmp[posCol, posCol])
        data[1, 2] <- sum(tmp[posCol, negCol])
        data[2, 1] <- sum(tmp[negCol, posCol])
        data[2, 2] <- sum(tmp[negCol, negCol])
        data <- as.table(data)
        positive <- "pos"
        rm(tmp)
    }
    negative <- colnames(data)[colnames(data) != positive]
    if (is.null(prevalence)) 
        prevalence <- sum(data[, positive])/sum(data)
    sens <- sensitivity(data, positive)
    spec <- specificity(data, negative)
    (sens * prevalence)/((sens * prevalence) + ((1 - spec) * 
                                                    (1 - prevalence)))
}

negPredValue.table <- function (data, negative = rownames(data)[-1], prevalence = NULL, 
                                ...) 
{
    if (!all.equal(nrow(data), ncol(data))) 
        stop("the table must have nrow = ncol")
    if (!all.equal(rownames(data), colnames(data))) 
        stop("the table must the same groups in the same order")
    if (nrow(data) > 2) {
        tmp <- data
        data <- matrix(NA, 2, 2)
        colnames(data) <- rownames(data) <- c("pos", "neg")
        negCol <- which(colnames(tmp) %in% negative)
        posCol <- which(!(colnames(tmp) %in% negative))
        data[1, 1] <- sum(tmp[posCol, posCol])
        data[1, 2] <- sum(tmp[posCol, negCol])
        data[2, 1] <- sum(tmp[negCol, posCol])
        data[2, 2] <- sum(tmp[negCol, negCol])
        data <- as.table(data)
        negative <- "neg"
        rm(tmp)
    }
    positive <- colnames(data)[colnames(data) != negative]
    if (is.null(prevalence)) 
        prevalence <- sum(data[, positive])/sum(data)
    sens <- sensitivity(data, positive)
    spec <- specificity(data, negative)
    (spec * (1 - prevalence))/(((1 - sens) * prevalence) + ((spec) * 
                                                                (1 - prevalence)))
}



sensitivity.default <- function (data, reference, positive = levels(reference)[1], na.rm = TRUE, 
                                 ...) 
{
    if (!is.factor(reference) | !is.factor(data)) 
        stop("inputs must be factors")
    if (length(unique(c(levels(reference), levels(data)))) != 
            2) 
        stop("input data must have the same two levels")
    if (na.rm) {
        cc <- complete.cases(data) & complete.cases(reference)
        if (any(!cc)) {
            data <- data[cc]
            reference <- reference[cc]
        }
    }
    numer <- sum(data %in% positive & reference %in% positive)
    denom <- sum(reference %in% positive)
    sens <- ifelse(denom > 0, numer/denom, NA)
    sens
}

specificity.default <- function (data, reference, negative = levels(reference)[-1], 
                                 na.rm = TRUE, ...) 
{
    if (!is.factor(reference) | !is.factor(data)) 
        stop("input data must be a factor")
    if (length(unique(c(levels(reference), levels(data)))) != 
            2) 
        stop("input data must have the same two levels")
    if (na.rm) {
        cc <- complete.cases(data) & complete.cases(reference)
        if (any(!cc)) {
            data <- data[cc]
            reference <- reference[cc]
        }
    }
    numer <- sum(data %in% negative & reference %in% negative)
    denom <- sum(reference %in% negative)
    spec <- ifelse(denom > 0, numer/denom, NA)
    spec
}

#' @title Performance Metrics
#' @description Function to list avaialble model performance
#' metrics to tune a model.
#' @return A data.frame containing the metric codes
#' and descriptions
#' @export
metrics <- function(){
    out <- data.frame(
        codes = c("AUC", "Sens", "Spec", 
                  "PPV", "NPV", "F1-Score"),
        description = c("Area Under ROC Curve", "Sensitivity", "Specificity",
                        "Positive Predictive Value", 
                        "Negative Predictive Value", 
                        "Harmonic mean of Precision and Recall")
    )
}
