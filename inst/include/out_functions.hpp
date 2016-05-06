#ifndef OUT_FUNCTIONS_HPP
#define OUT_FUNCTIONS_HPP

#include <RcppArmadillo.h>

using namespace Rcpp;

inline
arma::mat output_func_linear ( arma::mat x ){
  return x;
}

inline
arma::mat output_deriv_func_linear ( arma::mat x ){
  arma::mat ret(x.n_rows, x.n_cols);
  ret.fill(1);
  return ret;
}

inline
arma::mat output_func_softmax ( arma::mat x ){
// for multinomial problems
}

inline
arma::mat output_deriv_func_softmax ( arma::mat x ){
// derivative for multinomial problems
}

#endif
