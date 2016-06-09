## ----setup, include=FALSE, cache=FALSE-----------------------------------
library(knitr)
opts_chunk$set(
concordance=TRUE
)

## ----gridSearch----------------------------------------------------------
library(HGTools)
grid <- denovo_neuralnet_grid(res = 3, 
                              act_fcts = c("logistic", "relu"), 
                              dropout=TRUE)
dim(grid)
head(grid)

## ----multiLayerGrid------------------------------------------------------
expand.grid(.hidden = paste(c(10,5,5), collapse=","),
            .threshold = c(1,5),
            .act_fcts = c("logistic", "relu"))

