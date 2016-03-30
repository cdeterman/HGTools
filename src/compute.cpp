// standard headers
//#include <cmath>    // std::isnan

// local headers
#include "misc_functions.hpp"
#include "act_functions.hpp"

// With RcppArmadillo you don't include Rcpp.h
#include <RcppArmadillo.h>

// Enable C++11 via this plugin
// [[Rcpp::plugins(cpp11)]]

// include armadillo
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

//[[Rcpp::export]]
List c_compute(
  List nn, 
  NumericMatrix covariate_in) 
  {
//    cout << "called c_compute" << endl;
    // convert to arma::mat
    int n = covariate_in.nrow(), k = covariate_in.ncol();
    // create armadillo matrix, reuse memory
    arma::mat covariate_arma(covariate_in.begin(), n, k, false);

//    int c_rep = as<int>(rep);
    bool linear_output = as<bool>(nn["linear.output"]);
    List weights = nn["weights"];
//    List weights = full_weights[c_rep];
    
//    cout << "basic initializations" << endl;
    
    // Cannot inherit XPtr because not able to save
    // so it must be a string
    String act_fct = nn["act.fct"];
    
    // declare activation function
    XPtr<nmfptr> xptr_act_fct = act_func(act_fct);
    nmfptr c_act_fct = *xptr_act_fct;
    
//    cout << "xptrs complte" << endl;
    
    // declare integer vectors
    int length_weights = weights.size();
    arma::ivec nrow_weights = zeros<ivec>(length_weights);
    arma::ivec ncol_weights = zeros<ivec>(length_weights);
//    cout << "int vectors" << endl;    

    for (int i = 0; i < length_weights; i++){
      NumericMatrix tmp = as<NumericMatrix>(weights[i]);
      int tmp_nrow = tmp.nrow();
      int tmp_ncol = tmp.ncol();
      nrow_weights[i] = tmp_nrow;
      ncol_weights[i] = tmp_ncol;
    }

//    cout << "passed all initializations" << endl;
    
    arma::mat tmp_ones = ones<arma::mat>(covariate_arma.n_rows, 1);
    arma::mat covariate = join_rows(tmp_ones, covariate_arma);
    
//    cout << "added intercept" << endl;
    
    List neurons(length_weights);
    neurons[0] = covariate;
    
    if (length_weights > 1) {
      for (int i=0; i < (length_weights - 1); i++) {
        arma::mat temp = as<arma::mat>(neurons[i]) * 
                        as<arma::mat>(weights[i]);
        arma::mat act_temp = c_act_fct(temp);
        arma::mat tmp_ones = ones<arma::mat>(act_temp.n_rows, 1);
        neurons[i+1] = join_rows(tmp_ones, act_temp);
      }
    }
    
    arma::mat temp = as<arma::mat>(neurons[length_weights-1]) * 
                      as<arma::mat>(weights[length_weights-1]);
                     
    arma::mat net_result;
    if (linear_output) {
      net_result = temp;
    }else{
      net_result = c_act_fct(temp);
    } 
    
    // covert all neurons to SEXP objects
    for (int i=0; i < neurons.size(); i++){
        neurons[i] = wrap(neurons[i]);
    }
    
    return List::create(Named("neurons") = neurons,
                        Named("net.result") = wrap(net_result));
  }
