#ifndef ERR_FUNCTIONS_HPP
#define ERR_FUNCTIONS_HPP

#include <cmath>
#include <iostream>

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace std;

typedef arma::mat (*nmfptr2)(arma::mat x, arma::mat y);

inline
arma::mat err_ce ( arma::mat x, arma::mat y ){
  // Armadillo notation, % denotes element-wise multiplication
  arma::mat ret = -(y % log(x) + (1-y) % log(1-x));
  return ret;
}

inline
arma::mat err_deriv_ce ( arma::mat x, arma::mat y){
  arma::mat ret = (1-y)/(1-x) - y/x;
  return ret;
}

inline
arma::mat err_sse ( arma::mat x, arma::mat y ){
  arma::mat ret = (1/2) * pow((y-x),2);
  return ret;
}

inline
arma::mat err_deriv_sse ( arma::mat x, arma::mat y ){
  arma::mat ret = x-y;
  return ret;
}

inline
arma::mat err_deriv_ce_log ( arma::mat x, arma::mat y ){
  // Armadillo notation, % denotes element-wise multiplication
  arma::mat ret = (x % (1 - y)) - (y % (1 - x));
  return ret;
}

// function pointers
inline
XPtr<nmfptr2> err_func (String c , arma::mat response) {
  //NumericMatrix nm_response = DFtoNM(response);
  
  if ( c == "ce" ){
    LogicalVector binary_test;
    for ( unsigned int i = 0; i < response.n_cols; i ++){
      //NumericVector tmp = nm_response(_, i);
      arma::vec tmp = response.col(i);
      
      for( double* it=tmp.begin(); it != tmp.end(); ++it){
        if(*it == 0 || *it == 1){
          binary_test.push_back(1);
        }else{
          binary_test.push_back(0);
        }
      }
      
//      for ( int i = 0; i < nm_response.ncol(); i ++){
//      NumericVector tmp = nm_response(_, i);
//      
//      for( double* it=tmp.begin(); it != tmp.end(); ++it){
//        if(*it == 0 || *it == 1){
//          binary_test.push_back(1);
//        }else{
//          binary_test.push_back(0);
//        }
//      }
      
//      if(all_of(tmp.begin(), tmp.end(), [] (double value){
//        return value == 0 || value == 1;
//      })){
//        binary_test.push_back(1);
//      }else{
//        binary_test.push_back(0);
//      }
    }
    //if ( all_of(begin()))
    if ( is_true(all( binary_test )) ) {
      return(XPtr<nmfptr2>(new nmfptr2(&err_ce)));
    }else{
      cout << "'err.fct' was automaticaly set to sum of squared error (sse), because response isn't binary" << endl;
      return(XPtr<nmfptr2>(new nmfptr2(&err_sse)));
    }
  } else if ( c == "sse" ){
    return(XPtr<nmfptr2>(new nmfptr2(&err_sse)));
  } else {
    cout << "'err.fct' option not recognized." << endl;
    return XPtr<nmfptr2>(R_NilValue);
    //return 0;
  }
}

inline
XPtr<nmfptr2> err_deriv_func (String c , arma::mat response ) {
  //NumericMatrix nm_response = DFtoNM(response);
  
  if ( c == "ce" ){
    LogicalVector binary_test;
    for ( unsigned int i = 0; i < response.n_cols; i ++){
      //NumericVector tmp = nm_response(_, i);
      arma::vec tmp = response.col(i);
      
      for( double* it=tmp.begin(); it != tmp.end(); ++it){
        if(*it == 0 || *it == 1){
          binary_test.push_back(1);
        }else{
          binary_test.push_back(0);
        }
      }
      
//    for ( int i = 0; i < nm_response.ncol(); i ++){
//      NumericVector tmp = nm_response(_, i);
//      for( double* it=tmp.begin(); it != tmp.end(); ++it){
//        if(*it == 0 || *it == 1){
//          binary_test.push_back(1);
//        }else{
//          binary_test.push_back(0);
//        }
//      }
    }
    
    if ( is_true(all( binary_test )) ) {
      return(XPtr<nmfptr2>(new nmfptr2(&err_deriv_ce)));
    }else{
      cout << "'err.fct' was automaticaly set to sum of squared error (sse), because response isn't binary" << endl;
      return(XPtr<nmfptr2>(new nmfptr2(&err_deriv_sse)));
    }
  } else if ( c == "sse" ){
    return(XPtr<nmfptr2>(new nmfptr2(&err_deriv_sse)));
  } else if (c == "ce_log"){
    return(XPtr<nmfptr2>(new nmfptr2(&err_deriv_ce_log)));
  } else {
    cout << "'err.fct' option not recognized." << endl;
    return XPtr<nmfptr2>(R_NilValue);
    //return 0;
  }
}

#endif
