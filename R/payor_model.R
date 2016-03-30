#' @title Assign Testing Data Categories Manually
#' @description Takes the columns containing payor category identification in a test dataset
#' and returns a vector of the assigned category (e.g. 'C','M','D','S').
#' @param x A data.frame, matrix, or big.matrix that contains at required columns
#' that contain '0' or '1' to indicate presence of a category.  Designed to accept the
#' 'net.result' object from the \code{\link{compute}} function.
#' @details It is assumed that the specific order of the columns corresponds to a specific
#' order with respect to the specific \code{model_type}.
#' \describe{
#'  \item{payor}{Commerical, Medicare, Medicaid, and Self-Pay}
#' }
#' @return A character vector representing the four payor categories
# ' @export
assign_categories_manual <- function(x, model_type=NULL, dv=NULL){
  
    # checks for model_type
    assert_is_not_null(model_type)
    assert_is_character(model_type)
    if(is.null(dv) & model_type != "payor"){
        stop("You must provide the dependent variable name for
           non-payor type models")
    }
    if(model_type == "binary" & length(dv) > 1){
        stop("Too many dependent variable names provided.
             Binary models only accept one dv.")
    }
    
    stopifnot(length(model_type) == 1L)
  
    if(!is.null(model_type)){
        if(!model_type %in% model_types()[,1]){
            stop(paste("The 'model_type'", model_type, "is not recongized.
                        See model_types() for options.", sep=" "))
        }
    }
  
    switch(model_type,
           payor = {
               if(ncol(x) < 4){
                   stop("This function expects 4 columns for the 4 payor types.  
                  Your object does not contain 4 columns")
               }
               
               # paste the for columns together for indexing
               # Previously used do.call but only works well for data.frame
               # This allows for matrix & data.frame objects
               pc <- paste(x[,1],x[,2],x[,3],x[,4], sep="")
               
               pc[which(pc == "1000")] <- "C"
               pc[which(pc %in% c("0100", "1100", "1111", "0111", "0101", "0110"))] <- "M"
               pc[which(pc %in% c("0001", "0000", "0011", "1001", "1011", "1101"))] <- "S"
               pc[which(pc %in% c("0010", "1010", "1110"))] <- "D"
               
               return(pc)
           },
           binary = {
             idx <- c(round(x[,dv]))
             return(ifelse(idx, "1", "0"))
         }
     )
}


# @title Assign Predicted Payor Category
# @description This function takes the 'result.net' object from the 
# \code{\link{compute}} function 
# and returns a vector of the assigned payor category type.
# @param x The result.net object from \code{\link{compute}}.
# @return A character vector representing the four payor categories
# @export
assign_predicted_payor_category <- function(x){
  if(!is.matrix(x)){
    stop("Object 'x' is not a matrix.  Is this the 'result.net' object from neuralnet::compute?")
  }
  
  if(ncol(x) != 4){
    stop("This function expects 4 columns for the 4 payor types.  Your object does not 
         contain exactly 4 columns")
  }
  
  idx <- apply(x, 1, maxidx)
  prediction <- c('C', 'M', 'D', 'S')[idx]
  return(prediction)
  }

# 
# @title Assign Payor Model Confidence Categories and Scores
# @description Wrapper to quickly report the confidence categories and
# scaled scores for each sample from a neural net model.  This assigns scores
# based on the raw quantile level.
# @param nn.result The 'net.result' object returned from \code{\link{compute}}.
# @param raw_x The raw data initially submitted to \code{\link{neuralnet}}
# @param ntiles Numeric vector of manually defined quantile levels
# @param breaks Optional parameter to alter default quantile breaks
# @param cutoffs Optional parameter to alter default category breaks
# @return Matrix containing the four original category columns, raw activation values,
# calculated scores, and confidence categories
# @import foreach
# @export
assign_payor_scores_and_categories <- 
  function(nn.result, raw_x, ntiles=NULL, breaks=NULL, cutoffs=NULL, allowParallel=FALSE){
    
    assert_is_not_null(raw_x)
    
    if(!all(c("dv_had_c", "dv_had_m", "dv_had_d", "dv_had_s") %in% colnames(raw_x))){
      stop("'raw_x' doesn't contain the required columns 'dv_had_c', 'dv_had_m', 'dv_had_d', 'dv_had_s'
           Make sure raw_x contains the original classifications")
    }
    
    if(is.list(nn.result) && !is.data.frame(nn.result)){
      if(is.null(nn.result[["net.result"]])){
        stop("Submitted list does not contain net.result object.")
      }else{
        x <- nn.result$net.result
      }
    }else{
      if(ncol(nn.result) != 4L){
        stop("This function currently only defined for the four level payor model.
             The provided matrix does not contain 4 columns.")
      }else{
        x <- nn.result
      }
    }
    
    if(is.null(ntiles)){
      if(is.null(breaks)){
        breaks <- seq(from=0, to=1, by=0.001)
      }else{
        if(!is.numeric(breaks)){
          stop("'breaks' must be numeric.")
        }
        if(length(breaks) != 8L){
          stop("'breaks' must have 8 numeric elements.")
        }
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
      raw_x <- deepcopy(raw_x, cols=c("dv_had_c", "dv_had_m", "dv_had_d", "dv_had_s"))[,]
      colnames(raw_x) <- c("dv_had_c", "dv_had_m", "dv_had_d", "dv_had_s")
    }
    x <- data.frame(raw_x[,"dv_had_c"], raw_x[,"dv_had_m"], raw_x[,"dv_had_d"], raw_x[,"dv_had_s"],
                    x, 0, 0,0,0, 1,1,1,1)
    
    names(x) <- c('dv_had_c','dv_had_m', 'dv_had_d','dv_had_s', 
                  'raw_c','raw_m','raw_d','raw_s',
                  'score_c','score_m','score_d','score_s',
                  'cat_c','cat_m','cat_d','cat_s')
    
    # get quartiles parallelized
    #ntiles <- sapply(x[,5:8], quantile, probs=breaks)
    if(is.null(ntiles)){
      ntiles <- foreach(i = 5:8, .combine="cbind") %op% quantile(x[,i], probs=breaks)
    }
    
    # impute quantile scores
    x <- assign_quantile_scores(x, ntiles, allowParallel)
    
    # impute categories
    x <- assign_quantile_category(x, cutoffs)
    
    out <- new("payor_scores", x=x)
    return(out)
    }

payor_activation_scores_table <- 
  function(x, raw_x, d_var=NULL, step=0.01, breaks=NULL, allowParallel=FALSE){
    
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
    
    x <- cbind(x, 0, 0,0,0)
    
    names(x) <- c('raw_c','raw_m','raw_d','raw_s',
                  'score_c','score_m','score_d','score_s')
    
    # get quartiles parallelized
    ntiles <- foreach(i = 1:4, .combine="cbind") %op% quantile(x[,i], probs=breaks)
    
    nr <- nrow(x)
    nt <- nrow(ntiles)
    
    idx <- foreach(z=seq(nt-1)) %:%
      foreach(y=seq(4)) %op% {
        in_interval_int(x[,y], ntiles[z:(z+1),y])
      }
    
    # remove % symbol and convert to numeric for scores
    # scores <- as.numeric(gsub("%","",row.names(ntiles))) * 10
    scores <- as.numeric(gsub("%","",row.names(ntiles))) / 100
    
    
    for(i in seq(length(idx))){
      for(j in 1:4){
        cur_idx <- idx[[i]][[j]]
        x[cur_idx, j+4] <- scores[i]
      }
    }
    
    # fill the remaining max cases with 999
    idx_max <- lapply(seq(4), function(y) which(!seq(nr) %in% unlist(lapply(idx, function(x) x[[y]]))))
    for(i in seq(4)){
      x[idx_max[[i]], i+4] <- .999
    }
    
    return(x)
}

# @title Payor Model Score Subsetting
# @description Subset the output from scores_and_categories to evaluate
# prediction metrics above a given cutoff (e.g. 4, 7, etc.)
# @param x The output from \code{\link{assign_scores_and_categories}}
# @param cutoff The cutoff for filtering predictions (e.g. all those with confidence category >= 5)
# @param category Optional argument to filter only on one payor category
# @return \item{pred}{Subset of the neuralnet activations}
# @return \item{raw}{Subset of the test data payor category columns}
# @export
payor_score_subset <- function(x, cutoff, payor_category=NULL){
  
  assert_is_not_null(cutoff)
  
  if(!is.null(payor_category)){
    payor_category <- toupper(payor_category)
    
    if(length(payor_category) > 1){
      stop("Length of 'payor_category' cannot exceed 1.  Please set to NULL for filtering
           all categories or select from 'C','M','D',or 'S'.  NOT case sensitive.")
    }
    
    if(!payor_category %in% c('C','M','D','S')){
      stop("'payor_category' not recognized, please set to NULL for filtering
           all categories or select from 'C','M','D',or 'S'.  NOT case sensitive.")
    }
    }
  
  if(is.null(payor_category)){
    idx <- apply(sapply(13:16, function(y) x[,y]>=cutoff), 1, any)
  }else{
    idx <- switch(payor_category,
                  C = {x[,13] >= cutoff},
                  M = {x[,14] >= cutoff},
                  D = {x[,15] >= cutoff},
                  S = {x[,16] >= cutoff}
    )
  }
  pred <- x[idx,5:8]
  raw <- x[idx,1:4]
  #   out <- list(pred=as.matrix(pred), raw=raw)
  out <- new("payor_subset", pred=as.matrix(pred), raw=raw)
  return(out)
    }


#' @title Create Classification Curve for Payor Model
#' @description This function quickly creates the lift curves
#' for the payor model corresponding to the probability of correctly
#' classified payor category with respect to quantile levels.
#' @param x The output from \code{\link{percentile_profile}}
#' @param filename The name of the file to be saved as a pdf
#' @importFrom reshape2 melt
#' @import ggplot2
#' @export
payor_classification_curve <- function(x, filename){
  df <- as.data.frame(x)
  names(df) <- c('c','m','d','s','quantile')
  
  # reshape to long format for ggplot
  df2 <- melt(df, id.vars=c("quantile"))
  
  lift <- ggplot(df2, aes(x=quantile, y=value, group=variable)) + 
    geom_line(aes(colour=variable)) +
    ggtitle("Correct Probability by Raw Percentile") + 
    scale_x_reverse(limits=c(1,0)) + 
    scale_y_continuous(limits=c(0,1), breaks=seq(0,1,by=0.1)) +
    xlab("Percentile of Raw Score") +
    ylab("Probability of Correct Payor Type")
  ggsave(filename=paste(filename, "pdf", sep="."), lift)
}

#############################
### Demographic Profiling ###
#############################

demographic_profile_payor <- function(x, data){
  
  #assert_is_character(x)
  
  # make sure payor data
  # although strict to require certain columns it avoids the possibility
  # of passing data that is not 'payor' data.
  # current structure has first five columns as possible dependent variables
  # therefore verify names and then ignore
  # not including 'other' as hasn't historically been used but made sure to 
  # omit with subset
  col.names <- colnames(data)
  if(any(!c("dv_had_c", "dv_had_m", "dv_had_d", "dv_had_s") %in% col.names)){
    stop("The dataset does not appear to be 'payor' data.  Expected to see the 
         column names 'dv_had_c', 'dv_had_m', 'dv_had_d', and 'dv_had_s'.")
  }else{
    idx <- which(col.names %in% c("dv_had_c", "dv_had_m", "dv_had_d", "dv_had_s", "dv_had_o"))
    data <- data[,-c(idx)]
  }
  
  # split out payor categories
  commercial <- x[,grep("_c", colnames(x))]
  medicare <- x[,grep("_m", colnames(x))]
  medicaid <- x[,grep("_d", colnames(x))]
  self <- x[,grep("_s", colnames(x))]
  
  # collate into a list
  payor_list <- list(commercial, medicare, medicaid, self)
  
  # apply demographic profile to each category
  # add payor specific names
  payor_demog <- lapply(payor_list, function(x) demographic_profile_indv(x, data))
  payor_out <- do.call("cbind", payor_demog)
  colnames(payor_out) <- 
    paste(
      rep(c("commercial","medicare","medicaid","self"), each=7), 
      rep(seq(7), 4), 
      sep="_")
  row.names(payor_out) <- c("Sample Size", col.names[-idx])
  return(payor_out)
}

#' @title Assign Manual Payor Quantile Scores
#' @description This function computes scores defined by provided cut points.
#' This functions requires the user to provide the breakpoints to manually
#' calculate the scores.  This is representative of the original approach to
#' calculating scores.
#' @param x An n x 3 matrix containing the raw activation values from 
#' @param ntiles The n-tiles for each confidence category activation values
#' @param allowParallel Boolean indication if parallelization should be used
#' @return The 'x' matrix with the score columns filled
#' @export
assign_manual_payor_quantile_scores <- function(x, ntiles, allowParallel=FALSE){  
    # specify parallel
    `%op%` <- if(allowParallel){
        `%dopar%`
    }else{
        `%do%`
    }
    
    assert_is_not_null(ntiles)
    assert_is_list(ntiles)
    assert_is_of_length(ntiles, 4)
    
    for(i in 1:length(ntiles)){
        if(!is.matrix(ntiles[[i]])){
            if(length(ntiles[[i]]) != 8L && length(ntiles[[i]]) != 1001L){
                stop("Breakpoints must be either length of 8 or 1001")
            }
            ntiles[[i]] <- matrix(ntiles[[i]], ncol=1)
        }else{
            if(length(ntiles[[i]]) != 8L && length(ntiles[[i]]) != 1001L){
                stop("Breakpoints must be either length of 8 or 1001")
            }
        }
    }
    
    
    nc <- ncol(x)
    if(nc != 12){
        stop("Function only valid for the payor matrix which contains 12 columns")
    }
    
    ng <- 4
    nr <- nrow(x)
    nt <- unique(unlist(lapply(ntiles, nrow)))
    if(length(nt) > 1){
        stop("all breakpoints should be the same length")
    }
    
    # get indices of each score block
    idx <- foreach(i = seq(ng)) %:%
        foreach(t = seq(nt-1)) %op% {
            in_interval_int(as.numeric(x[,i]), ntiles[[i]][(t):(t+1),])
        }   
    
    if(nt == 1001){
        for(i in seq(ng)){
            for(t in seq(nt-1)){
                cur_idx <- idx[[i]][[t]]
                x[cur_idx, i+ng] <- t
            }
        }
    }else{
        score_bases <- c(560, 560, 810, 910, 960, 985, 995)
        score_adds <- c(1, 250, 100, 50, 25, 10, 5)
        
        for(i in seq(ng)){
            for(t in seq(nt-1)){
                cur_idx <- idx[[i]][[t]]
                x[cur_idx, i+ng] <- floor(score_bases[t] + score_adds[t] * (x[cur_idx,i] - ntiles[[i]][t])/(ntiles[[i]][t+1] - ntiles[[i]][t]))
            }
        }
    }
    
    idx_max <- list()
    idx_min <- list()
    for(i in seq(ng)){
        idx_max[i] <- list(which(x[,i] > ntiles[[i]][1000]))
        idx_min[i] <- list(which(x[,i] < ntiles[[i]][1]))
    }
    
    
    # fill the remaining max cases with 999
    #     idx_max <- lapply(seq(ng), 
    #                       function(y){
    #                           which(!seq(nr) %in% unlist(idx[[y]]))
    #                       }
    #     )
    
    # fill the remaining max cases with 999 and min with 0
    for(i in seq(ng)){
        x[idx_max[[i]], i+ng] <- 999
        x[idx_min[[i]], i+ng] <- 0
    }
    
    return(x)     
}

#' @export
assign_manual_payor_quantile_categories <- function(x, cutoffs=NULL){  
    
    # set vector for categories
    categories <- seq(7)
    # set vector for score cutoffs
    if(is.null(cutoffs)){
        cutoffs <- c(0,560,810,910,960,985,995,999)
    }else{
        stopifnot(length(cutoffs) == 8L)
        assert_all_are_positive(cutoffs)
        if(is.numeric(cutoffs) | is.integer(cutoffs)){
            stop("'cutoffs' is not of type 'integer' or 'numeric'")
        }
    }
    
    nc <- ncol(x)
    ng <- 4
    
    # get indices for categories
    idx <- lapply(categories, function(z) 
        lapply(seq(ng), function(y) 
            in_interval_int(x[,y+ng], cutoffs[z:(z+1)])))
    
    # assign categories
    for(i in categories){
        for(j in (nc-ng+1):nc){
            cur_idx <- idx[[i]][[j-(nc-ng)]]
            x[cur_idx, j] <- categories[i]
        }
    }
    
    
    out <- as.matrix(x)
    return(out)
}


