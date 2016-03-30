

#' @import foreach
assign_binary_scores_and_categories <- 
  function(nn.result, raw_x, dvs, 
           ntiles=NULL, breaks=NULL, cutoffs=NULL, 
           allowParallel=FALSE){
    
    assert_is_not_null(raw_x)
    assert_is_character(dvs)
    
    if(is.list(nn.result) && !is.data.frame(nn.result)){
      if(is.null(nn.result[["net.result"]])){
        stop("Submitted list does not contain net.result object.")
      }else{
        x <- nn.result$net.result
      }
    }else{
      if(ncol(nn.result) > 1L){
        stop("This function currently only defined for a binary model.
             The provided matrix contains more than 1 column.")
      }else{
        x <- nn.result
      }
    }
    
    if(is.null(ntiles)){
      if(is.null(breaks)){
        breaks <- seq(from=0, to=1, by=0.001)
      }else{
          assert_is_numeric(breaks)
          assert_is_of_length(breaks, 8L)
      }
    }
    
    
    # specify parallel
    `%op%` <- if(allowParallel){
      `%dopar%`
    }else{
      `%do%`
    }
    
    # expand data.frame to contain results
    if(is.big.matrix(raw_x)){
        # subset and covert to matrix
        raw_x <- matrix(deepcopy(raw_x, cols=dvs)[,], ncol=1)
        colnames(raw_x) <- dvs
    }
    
    x <- data.frame(raw_x[,dvs], x, 0, 1)
    names(x) <- paste(c('dv_', 'raw_', 'score_', 'cat_'), 
                      dvs, sep="")

    
    
    # get quantiles
    if(is.null(ntiles)){
        ntiles <- as.matrix(quantile(x[,2], probs=breaks)) 
    }
    
    # impute quantile scores
    x <- assign_quantile_scores(x, ntiles, allowParallel)
    
    # impute categories
    x <- assign_quantile_category(x, cutoffs)
    
    out <- new("binary_scores", x=x)
    return(out)
    }




binary_activation_scores_table <- 
  function(x, raw_x, d_var, step=0.01, breaks=NULL, allowParallel=FALSE){
    
#     x <- net.result
    
    if(is.null(breaks)){
      breaks <- seq(from=0, to=1, by=step)
    }else{
      if(!is.numeric(breaks)){
        stop("'breaks' must be numeric.")
      }
    }
    
    
    # specify parallel
    `%op%` <- if(allowParallel){
      `%dopar%`
    }else{
      `%do%`
    }
    
    x <- cbind(x, 0)
    
    names(x) <- c(d_var, 'score')
    
    # get quartiles parallelized
    ntiles <- quantile(x[,1], probs=breaks)
    
    nr <- nrow(x)
    nt <- length(ntiles)
    
    idx <- foreach(z=seq(nt-1)) %op%
        in_interval_int(x[,1], ntiles[z:(z+1)])
    
    # remove % symbol and convert to numeric for scores
    # scores <- as.numeric(gsub("%","",row.names(ntiles))) * 10
    scores <- as.numeric(gsub("%","",names(ntiles))) / 100
    
    
    for(i in seq(length(idx))){
        cur_idx <- idx[[i]]
        x[cur_idx, 2] <- scores[i]
    }
    
    # fill the remaining max cases with 999
    idx_max <- lapply(1, function(y) which(!seq(nr) %in% unlist(lapply(idx, function(x) x[[y]]))))
    for(i in 1){
      x[idx_max[[i]], 2] <- .999
    }
    
    return(x)
  }

#' @title Binary Model Score Subsetting
#' @description Subset the output from scores_and_categories to evaluate
#' prediction metrics above a given cutoff (e.g. 4, 7, etc.).  This is for the
#' generic binary model with no customizations.
#' @param x The output from \code{\link{assign_scores_and_categories}} of class
#' \code{binary_scores}.
#' @param cutoff The cutoff for filtering predictions (e.g. all those 
#' with confidence category >= 5)
#' @return \item{pred}{Subset of the neuralnet activations}
#' @return \item{raw}{Subset of the test data binary category}
binary_score_subset <- function(x, cutoff){
  
  assert_is_not_null(cutoff)
  assert_is_in_closed_range(cutoff, 1, 7)
  
  # get row indices
  idx <- x[,4] >= cutoff  
  
  # pull specific rows
  pred <- x[idx,2]
  raw <- x[idx,1]
  out <- new("binary_subset", pred=as.matrix(pred), raw=as.matrix(raw))
  return(out)
}


#' @title Assign Predicted Binary Category
#' @description This function takes the 'result.net' object from the 
#' \code{\link{compute}} function 
#' and returns a vector of the assigned binary category codes 
#' (i.e. present/absent).
#' @param x The result.net object from \code{\link{compute}}.
#' @return A character vector representing the binary categories
# @export
assign_predicted_binary_category <- function(x){
    if(!is.matrix(x)){
        stop("Object 'x' is not a matrix.  Is this the 'result.net' object from neuralnet::compute?")
    }
    
    if(ncol(x) > 1){
        stop("This function expects only 1 column.  Your object does not 
         contain exactly 1 column")
    }
    
    idx <- c(round(x))
    prediction <- ifelse(idx == 1, '1', '0')
    return(prediction)
}


#' @importFrom caTools colAUC
auc_binary <- function(x, dv){
    
    raw_dv <- paste("dv", dv, sep="_")
    pred_dv <- paste("raw", dv, sep="_")
    
    if(!pred_dv %in% colnames(x)){
        print("the given pred_dv")
        print(pred_dv)
        print("available columns are:")
        print(colnames(x))
        stop("predicted variable not in matrix")
    }
    
    if(!raw_dv %in% colnames(x)){
        stop("raw variable not in matrix")
    }
    
    pred <- ifelse(c(round(x[,pred_dv])), 1, 0)
    orig <- ifelse(c(x[,raw_dv]), 1, 0)
    
    tmp.auc <- colMeans(colAUC(pred, orig,
                               plotROC=FALSE, alg="ROC"))
    return(tmp.auc)
}

auc_sub_binary <- function(x){
    
    pred <- ifelse(c(round(x@pred)), 1, 0)
    orig <- ifelse(c(x@raw), 1, 0)
    
    tmp.auc <- colMeans(colAUC(pred, orig,
                               plotROC=FALSE, alg="ROC"))
    return(tmp.auc)
}


#' @title Create Classification Curve for Binary Model
#' @description This function quickly creates the classification curves
#' for a generic binary model corresponding to the probability of correctly
#' classified category with respect to quantile levels.
#' @param x The output from \code{\link{percentile_profile}}
#' @param filename The name of the file to be saved as a pdf
#' @importFrom reshape2 melt
#' @import ggplot2
binary_classification_curve <- function(x, filename){
  
  assert_is_character(filename)
  
  df <- as.data.frame(x)
  names(df) <- c('Category','quantile')
  
  # reshape to long format for ggplot
  df2 <- melt(df, id.vars=c("quantile"))
  
  lift <- ggplot(df2, aes(x=quantile, y=value, group=variable)) + 
    geom_line(aes(colour=variable)) +
    ggtitle("Correct Probability by Raw Percentile") + 
    scale_x_reverse(limits=c(1,0), breaks=seq(0,1, by=.1)) + 
    scale_y_continuous(limits=c(0,1), breaks=seq(0,1, by=.1)) +
    xlab("Percentile of Raw Score") +
    ylab("Probability of Correct Category")
  ggsave(filename=paste(filename, "pdf", sep="."), lift)
}


#' @title Create Lift Curve for Binary Model
#' @description This function quickly creates the lift curves
#' for a generic binary model corresponding to the probability of correctly
#' classified category with respect to quantile levels.
#' @param x The output from \code{\link{percentile_profile}}
#' @param data A data.frame or matrix containing the original dependent
#' variable column
#' @param dvs A string specifying the dependent variable
#' @param filename The name of the file to be saved as a pdf
#' @importFrom reshape2 melt
#' @import ggplot2
binary_lift_curve <- function(x, data, dvs, filename){
  
  assert_is_character(dvs)
  assert_is_character(filename)
  
  df <- as.data.frame(x)
  names(df) <- c('Category','quantile')
  
  # reshape to long format for ggplot
  df2 <- melt(df, id.vars=c("quantile"))
  utilization = sum(data[,dvs])/nrow(data)
  df2$value <- df2$value/utilization
  
  y_max = ceiling(max(df2$value/utilization))
  
  lift <- ggplot(df2, aes(x=quantile, y=value, group=variable)) + 
    geom_line(aes(colour=variable)) +
    ggtitle("Lift Curve") + 
    scale_x_reverse(limits=c(1,0), breaks=seq(0,1,by=.1)) + 
    scale_y_continuous(breaks=seq(0,y_max, by=.1)) +
    xlab("Percentile of Raw Score") +
    ylab("Lift")
  ggsave(filename=paste(filename, "pdf", sep="."), lift)
}

binary_gains_chart <- function(x, filename){
  
  assert_is_character(filename)
  
  df <- as.data.frame(x)
  names(df) <- c('Category','quantile')
  
  # reshape to long format for ggplot
  df2 <- melt(df, id.vars=c("quantile"))
  
  df2$count <- cumsum(df2$value * nrow(df2))/sum(df2$value * nrow(df2))
  
  out <- ggplot(df2, aes(x=quantile, y=count, group=variable)) + 
    geom_line(aes(colour=variable)) +
    ggtitle("Gains Chart") + 
    scale_x_reverse(limits=c(1,0), breaks=seq(0,1,by=.1)) + 
    scale_y_continuous(breaks=seq(0,1, by=.1)) +
    xlab("Percentile of Raw Score") +
    ylab("Percent Classified")
  ggsave(filename=paste(filename, "pdf", sep="."), out)
}


#############################
### Demographic Profiling ###
#############################

demographic_profile_binary <- function(x, data, dv_name, categories=TRUE){
    
    assert_is_character(dv_name)
    col.names <- colnames(data)
    
    idx <- which(col.names == "demo_gender_f")
    data <- data[,idx:ncol(data)]
    
    # split out categories
    if(categories){
        ids <- matrix(x[, paste("cat", dv_name, sep="_")], ncol=1)
        newCols <- paste(
            rep(c(dv_name), each=7), 
            seq(7), 
            sep="_")
    }else{
        ids <- matrix(x[, paste("score", dv_name, sep="_")], ncol=1)
        newCols <- paste(
            rep(c(dv_name), each=1000), 
            seq(from = 0, to = 999), 
            sep="_")
    }
    
    # collate into a list
    binary_list <- list(ids)
    
    # apply demographic profile to each category
    # add payor specific names
    if(categories){
        binary_demog <- lapply(binary_list, function(x) demographic_profile_indv(x, data))
    }else{
        binary_demog <- lapply(binary_list, function(x) demographic_profile_indv2(x, data))
    }
    
    binary_out <- do.call("cbind", binary_demog)
    
    colnames(binary_out) <- newCols
        
    row.names(binary_out) <- c("Sample Size", colnames(data))
    return(binary_out)
}


extract_binary_breakpoints <- function(x){
    # convert to data.frame for aggregate function
    df <- as.data.frame(x)
    agg <- aggregate(df, by = list(category = df[,4]), max)
    breaks <- c(0, agg[2:6,3], 1)
    return(breaks)
}
