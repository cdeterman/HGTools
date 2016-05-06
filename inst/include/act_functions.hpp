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
arma::mat act_relu( arma::mat x ){
    // relu = max(0, x)
    
    // find negatives
    arma::uvec negs = arma::find(x < 0);
    arma::mat ret = x;
    
    // change those indices to 0
    ret.elem(negs).zeros();
    return ret;
}

inline
arma::mat act_deriv_relu( arma::mat x ){
    
    // find positives
    arma::uvec pos = arma::find(x >= 0);
    arma::uvec negs = arma::find(x < 0);
    
    // assign 1 for >= 0
    // assign 0 for < 0
    arma::mat ret = x;
    ret.elem(pos).ones();
    ret.elem(negs).zeros();
    return ret;
}

inline
XPtr<nmfptr> act_func( String c ) {
  if (c == "tanh"){
    return(XPtr<nmfptr>(new nmfptr(&act_tanh)));
  } else if ( c == "logistic" ){
    return(XPtr<nmfptr>(new nmfptr(&act_logistic)));
  } else if (c == "relu" ){
    return(XPtr<nmfptr>(new nmfptr(&act_relu)));
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
  } else if ( c == "relu" ){
    return(XPtr<nmfptr>(new nmfptr(&act_deriv_relu)));
  } else {
    std::cout << "'act.deriv.fct' option not recognized." << std::endl;
    return XPtr<nmfptr>(R_NilValue);
    //return 0;
  }
}

#endif // ACT_FUNCTIONS
