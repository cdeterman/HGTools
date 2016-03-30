#ifndef ACT_FUNCTIONS_HPP
#define ACT_FUNCTIONS_HPP

#include <cmath>      // pow, tanh, exp
#include <iostream>

#include <RcppArmadillo.h>

using namespace Rcpp;

typedef arma::mat (*nmfptr)(arma::mat x);

inline
arma::mat act_tanh( arma::mat x ){
  return tanh(x);
}

inline
arma::mat act_deriv_tanh( arma::mat x ){
  arma::mat ret = 1-pow(x, 2);
  return ret;
}

inline 
arma::mat act_logistic( arma::mat x ){
  arma::mat ret = 1/(1+exp(-x));
  return ret;
}

inline
arma::mat act_deriv_logistic( arma::mat x ){
  // Armadillo notation, % denotes element-wise multiplication
  arma::mat ret = x % ( 1-x );
  return ret;
}

inline
XPtr<nmfptr> act_func( String c ) {
  if (c == "tanh"){
    return(XPtr<nmfptr>(new nmfptr(&act_tanh)));
  } else if ( c == "logistic" ){
    return(XPtr<nmfptr>(new nmfptr(&act_logistic)));
  } else {
    std::cout << "'act.fct' option not recognized." << std::endl;
    return XPtr<nmfptr>(R_NilValue);
//    return 0;
  }
}

inline
XPtr<nmfptr> act_deriv_func( String c ) {
  if (c == "tanh"){
    return(XPtr<nmfptr>(new nmfptr(&act_deriv_tanh)));
  } else if ( c == "logistic" ){
    return(XPtr<nmfptr>(new nmfptr(&act_deriv_logistic)));
  } else {
    std::cout << "'act.deriv.fct' option not recognized." << std::endl;
    return XPtr<nmfptr>(R_NilValue);
    //return 0;
  }
}

#endif // ACT_FUNCTIONS
