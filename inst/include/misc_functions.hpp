#ifndef MISC_FUNCTIONS_HPP
#define MISC_FUNCTIONS_HPP

#include <RcppArmadillo.h>

using namespace Rcpp;

inline
NumericMatrix DFtoNM(DataFrame x) {
  int nRows = x.nrows();
  NumericMatrix y(nRows, x.size());
  for (int i=0; i<x.size(); i++){
    y(_,i)=NumericVector(x[i]);
  }
  return y;
}


inline 
arma::vec c_unlist(List x)
{
  arma::vec x_vec = vectorise(as<arma::mat>(x[0]));
  for(int i=1; i < x.size(); i++){
    arma::vec tmp_vec = vectorise(as<arma::mat>(x[i]));
    x_vec = join_cols(x_vec, tmp_vec);
  }
  return x_vec;
}


inline 
List c_relist( 
  arma::vec x,
  arma::ivec nrow,
  arma::ivec ncol
  )
  {    
    List list_x(nrow.n_elem);
    int length;
    int start=0;
    int end;
    for (unsigned int w=0; w < nrow.n_elem; w++){
        length = nrow[w] * ncol[w];
        end = start + length - 1;
        
        arma::vec tmp_vec = x.subvec(start, end);
        arma::mat tmp_mat = reshape(tmp_vec, nrow[w], ncol[w]);
        list_x[w] = tmp_mat;
        
        start += length;
    }
    return wrap(list_x);
  }
  
#endif // MISC_FUNCTIONS_HPP
