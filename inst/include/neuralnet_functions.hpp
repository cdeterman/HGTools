#ifndef NEURALNET_FUNCTIONS_HPP
#define NEURALNET_FUNCTIONS_HPP

#include <iostream>
#include <cstddef>
#include <cmath>

#ifndef nullptr
#define nullptr NULL
#endif

#include <RcppArmadillo.h>

// other local headers
#include "misc_functions.hpp"
#include "act_functions.hpp"
#include "err_functions.hpp"
#include "out_functions.hpp"

using namespace Rcpp;
using namespace arma;

inline
List c_generate_startweights(
  List model_list, 
  IntegerVector hidden, 
  SEXP startweights, 
  int rep, 
  SEXP exclude, 
  SEXP constant_weights) 
  {
    //cout << "called generate_startweights" << endl;
    
    int length_weights;
    
    IntegerVector dx_exclude;
    IntegerVector dx_constant_weights;
    NumericVector dx_startweights;
    
    if(!Rf_isNull( exclude )){
      dx_exclude = as<IntegerVector>(exclude);
    }
    if(!Rf_isNull( startweights )){
      dx_startweights = as<NumericVector>(startweights);
    }
    if(!Rf_isNull( constant_weights )){
      dx_constant_weights = as<NumericVector>(constant_weights);
    }
   
    // current problem, nrow/ncol_weights length is dependent upon hidden
    // Need to conditionally set the length of the array/vector
    IntegerVector nrow_weights;
    IntegerVector ncol_weights;
       
    vector< string > inputs = as<vector< string> >(model_list["variables"]);
    vector< string > outputs = as<vector< string> >(model_list["response"]);
    int input_count = inputs.size();
    int output_count = outputs.size();
    int length_hidden_nodes = hidden.size();
    
    // needs is_true around the Rcpp 'any' function for bool type
    if (!(length_hidden_nodes == 1 && is_true(any(hidden == 0)))) {
      
      length_weights = length_hidden_nodes + 1;
      
      nrow_weights.push_back(input_count + 1);
      ncol_weights.push_back(hidden[0]);
      
      if (length_hidden_nodes > 1) {
        for (int i = 1; i < length_hidden_nodes; i++) {
          nrow_weights.push_back(hidden[i - 1] + 1);
          ncol_weights.push_back(hidden[i]);
        }
      }
      
      // set -2 for 0-indexing
      nrow_weights.push_back(hidden[length_weights-2] + 1);
      ncol_weights.push_back(output_count);
    }
    else {
      length_weights = 1;
      nrow_weights.push_back(input_count + 1);
      ncol_weights.push_back(output_count);
    }
    
//    cout << "nrow.weights" << endl;
//    for(int i=0; i < nrow_weights.size(); i++){
//        cout << nrow_weights[i] << endl;
//    }
//
//    cout << "ncol.weights" << endl;
//    for(int i=0; i < ncol_weights.size(); i++){
//        cout << ncol_weights[i] << endl;
//    }
    
    //cout << "passed if else clause" << endl;
    
    IntegerVector tmp_vec = ncol_weights * nrow_weights;
    int length = accumulate(tmp_vec.begin(), tmp_vec.end(), 0);
//    cout << "length" << endl;
//    cout << length << endl;
    
    //cout << "passed accumulate" << endl;
    
    arma::vec vector_weights;
    vector_weights.set_size(length);
    
    length = length - dx_exclude.size();
    
    if (!Rf_isNull( exclude )) {
      cout << "custom 'exclude' not yet implemented";
      return 0;
//      if (Rf_isNull(startweights) || dx_startweights.size() < (length*rep)) {
//        for ( int i = 0; i < length; i++ ){
//          if( find(dx_exclude.begin(), dx_exclude.end(), i+1) != dx_exclude.end() ){
//            vector_weights[i] = rnorm(generator);
//          }
//        }
//        //vector[-exclude] <- rnorm(length);
//      }else{
//        cout << "custom startweights not yet implemented" << endl;
//        return 0;
//      } 
    }
    else {
      if (Rf_isNull(startweights) || dx_startweights.size() < (length*rep)) {
        vector_weights = as<arma::vec>(rnorm(length));        
        //vector[-exclude] <- rnorm(length);
      }else{
        cout << "custom startweights not yet implemented" << endl;
        return 0;
      } 
    }
    
//    cout << "vector" << endl;
//    cout << vector_weights << endl;
    
    //cout << "passed 'exclude' if else clause" << endl;
    
    if (!Rf_isNull( exclude ) && !Rf_isNull( constant_weights )) {
      if (dx_exclude.size() < dx_constant_weights.size()) {
        cout << "constant.weights contains more weights than exclude" << endl;
        return 0;
      }else{
        cout << "custom exclude not yet implemented" << endl;
        return 0;
      }
    }
    
//    List weights(nrow_weights.size());
//    arma::mat tmp;
//    int start = 0;
//    int end = nrow_weights[0];
//    for( int i=0; i<nrow_weights.size(); i++ ){
//
//      // Create arma matrix of specific dimensions
//      tmp.set_size(nrow_weights[i], ncol_weights[i]);
//      
//      // fill arma matrix by columns
//      for (unsigned int c = 0; c < tmp.n_cols; c++){
//        arma::vec sub = vector_weights.subvec(start, end-1);
//        //cout << "filling tmp matrix" << endl;
//        tmp.col(c) = sub;
//      }
//      //cout << "filled tmp matrix" << endl;
//            
//      weights[i] = tmp;
//      start += end;
//      end += nrow_weights[i+1];
//    }

    List weights = c_relist(vector_weights, as<arma::ivec>(wrap(nrow_weights)),
                          as<arma::ivec>(wrap(ncol_weights)));

    return List::create(Named("weights") = wrap(weights), 
                        Named("exclude") = wrap(exclude));
  }


inline
List c_compute_net(
  List weights, 
  int length_weights, 
  arma::mat covariate, // big.matrix
  SEXP act_fct, 
  SEXP act_deriv_fct, 
  SEXP output_act_fct, 
  SEXP output_act_deriv_fct, 
  bool special,
  bool dropout = true,
  double visible_dropout = 0,
  arma::vec hidden_dropout = arma::zeros<vec>(1)) 
  {
//    cout << "called c_compute_net" << endl;
    
//    std::cout << "check dropout" << std::endl;
//    std::cout << dropout << std::endl;
    
//    // convert DataFrame to arma::mat
//    // first convert to NumericMatrix
//    NumericMatrix nm_covariate = DFtoNM(covariate);
//    // now convert to arma::mat
//    int n = nm_covariate.nrow(), k = nm_covariate.ncol();
//    // create armadillo matrix, reuse memory
//    arma::mat covariate_arma(nm_covariate.begin(), n, k, false);
    
    // declare the function pointers
    XPtr<nmfptr> xptr_act_fct(act_fct);
    XPtr<nmfptr> xptr_act_deriv_fct(act_deriv_fct);
    XPtr<nmfptr> xptr_output_act_fct(output_act_fct);
    XPtr<nmfptr> xptr_output_act_deriv_fct(output_act_deriv_fct);
    
    //cout << "xptrs imported" << endl;
    
    // make pointers functional
    nmfptr c_act_fct = *xptr_act_fct;
    nmfptr c_act_deriv_fct = *xptr_act_deriv_fct;
    nmfptr c_output_act_fct = *xptr_output_act_fct;
    nmfptr c_output_act_deriv_fct = *xptr_output_act_deriv_fct;
    
    //cout << "xptrs now functional" << endl;
    
    List neuron_deriv(length_weights);
    
    List neurons(length_weights+1);
    neurons[0] = covariate;
    
//    std::cout << "visible" << std::endl;
//    std::cout << covariate.head_rows(10) << std::endl;
    
    // visible dropouts
    if(dropout){
        if(visible_dropout > 0){
            // generate random 0,1 for dropout
            // convert to Rcpp::NumericMatrix (fastest I have seen)
            NumericVector draws = rbinom(covariate.size(), 1, 1-visible_dropout);
            NumericMatrix binomMat = NumericMatrix(covariate.n_rows, covariate.n_cols, draws.begin());
            
            // convert to armadillo without copying memory 
            // for fast multiplication using whichever BLAS installed
            arma::mat armaBinomMat(binomMat.begin(), binomMat.rows(), binomMat.cols(), false);

            //std::cout << "binomial sample" << std::endl;
            //std::cout << armaBinomMat << std::endl;
            
            neurons[0] = covariate % armaBinomMat;
        }
    }
    
    {
//        std::cout << "masked visible" << std::endl;
//        arma::mat temp = neurons[0];
//        std::cout << temp.head_rows(10) << std::endl;
    }
    
    
    if (length_weights > 1) {
        for (int i=0; i < (length_weights - 1); i++) {
            arma::mat neurons_tmp = neurons[i];
            arma::mat weights_tmp = weights[i];
            arma::mat temp = neurons_tmp * weights_tmp;            
            arma::mat act_temp = c_act_fct(temp);
            
            arma::uvec negs = arma::find(act_temp < 0);
            
            
            if(negs.n_elem > 0){
                stop("shouldn't be any negative elements");
            }
            
//            std::cout << "activations" << std::endl;
//            std::cout << act_temp.head_rows(10) << std::endl;
            
            // hidden dropouts
            if(dropout){
                if(hidden_dropout[i] > 0){
                    // generate random 0,1 for dropout
                    // convert to Rcpp::NumericMatrix (fastest I have seen)
                    NumericVector draws = rbinom(act_temp.size(), 1, 1-hidden_dropout[i]);
                    NumericMatrix binomMat = NumericMatrix(act_temp.n_rows, act_temp.n_cols, draws.begin());
                    
                    // convert to armadillo without copying memory 
                    // for fast multiplication using whichever BLAS installed
                    arma::mat armaBinomMat(binomMat.begin(), binomMat.rows(), binomMat.cols(), false);
        
                    //std::cout << "binomial sample" << std::endl;
                    //std::cout << armaBinomMat << std::endl;
                    
                    act_temp = act_temp % armaBinomMat;
                }
            }
            
//            std::cout << "new activations" << std::endl;
//            std::cout << act_temp.head_rows(10) << std::endl;

            if (special) {
                neuron_deriv[i] = c_act_deriv_fct(act_temp);
            }else{
                neuron_deriv[i] = c_act_deriv_fct(temp);
            }
            arma::mat tmp_ones = arma::ones<arma::mat>(act_temp.n_rows, 1);
            neurons[i+1] = join_rows(tmp_ones, act_temp);
        }
    }
    
//    cout << "passed length_weights condition" << endl;
    
//    cout << "neurons list" << endl;
//    for(int i=0; i < neurons.size(); i++){
//      Rcout << as<arma::mat>(neurons[i]).submat(0,0,5,1);
//    }
    

    arma::mat neurons_temp = neurons[length_weights-1];
//    cout << "neurons_temp" << endl;
//    Rcout << neurons_temp.submat(0,0,5,1) << endl;
    //cout << "indexed neurons" << endl;
    arma::mat weights_temp = weights[length_weights-1];
    arma::mat temp = neurons_temp*weights_temp;
    
//    cout << "temp" << endl;
//    Rcout << temp.submat(0,0,5,0) << endl;
    
//    cout << "passed neuron arma matrices" << endl;
    
    arma::mat net_result = c_output_act_fct(temp);
    if (special) {
      neuron_deriv[length_weights-1] = c_output_act_deriv_fct(net_result);
//      cout << "net.result" << endl;
//      Rcout << net_result.submat(0,0,5,0) << endl;
    }else{
      neuron_deriv[length_weights-1] = c_output_act_deriv_fct(temp);
    }
    
//    std::cout << "got neuron derivs" << std::endl;
    
//    if (any(is.na(neuron.deriv))) {
//      stop("neuron derivatives contain a NA; varify that the derivative function does not divide by 0", 
//           call. = FALSE)
//    }

    return List::create(Named("neurons") = neurons, 
                        Named("neuron_deriv") = neuron_deriv,
                        Named("net_result") = net_result);
  }
 
 
// this function removes the first row (i.e. intercept) of a arma matrix
inline
arma::mat remove_intercept(arma::mat m)
{
//  cout << "called remove_intercept" << endl;
  int from = 1;
  int to = m.n_rows - 1;
  arma::mat out = m.rows(from, to);
  return out;
}
 

inline
arma::vec c_calculate_gradients(
    List weights, 
    int length_weights, 
    List neurons, 
    List neuron_deriv, 
    arma::mat err_deriv, 
    int exclude, 
    bool linear_output) 
    {
    //cout << "called calculate_gradients" << endl;
    
    // declare arma matrix
    arma::mat delta;
    
    if (!linear_output) {
        delta = as<arma::mat>(neuron_deriv[length_weights-1]) % err_deriv;
    }else{
        delta = err_deriv;
    } 
    
    arma::mat neuron = neurons[length_weights-1];
    
    arma::mat gradients = trans(neuron) * delta;
    arma::vec gradients_v = vectorise(gradients);

    if (length_weights > 1){
      // changed to -2 for 0-based c++
      for(int w=length_weights-1; w > 0; w--){
        arma::mat weight = remove_intercept(weights[w]);
        delta = as<arma::mat>(neuron_deriv[w-1]) % (delta * trans(weight));
        arma::mat neuron_delta = trans(as<arma::mat>(neurons[w-1])) * delta;
        gradients_v = join_cols(vectorise(neuron_delta), gradients_v);        
      }
    }
    
    // place to potentially deal with multiple excludes
//    if(as<arma::vec>(exclude){
//        // make sure sorted so subtraction will work appropriately
//        arma::vec excl = sort(as<arma::vec>(exclude));
//    }    
//        for( unsigned int i=0; i < excl.n_elem; i++){
//            gradients_v.shed_row(excl[i]-1);
//            excl = excl - 1;
//        }

    // exclude not currently implemented
    //gradients_v.shed_row(exclude-1);
    
    //cout << "finished calculate_gradients" << endl;
    
    return gradients_v;
  }  


inline
List c_plus(
  arma::vec gradients, 
  arma::vec gradients_old, 
  List weights, 
  arma::ivec nrow_weights, 
  arma::ivec ncol_weights, 
  arma::vec learningrate, 
  List learningrate_factor, 
  List learningrate_limit, 
  SEXP exclude) 
{
//    cout << "called c_plus" << endl;

    arma::vec weights_vec = c_unlist(weights);
    arma::vec sign_gradient = sign(gradients);
    arma::vec temp = gradients_old % sign_gradient;
    
//    cout << "original weights" << endl;
//    Rcout << weights_vec << endl;
    
    // declare unsigned vectors
    arma::uvec positive = (temp > 0);
    arma::uvec negative = (temp < 0);
    
    // get indices because 'which' function doesn't exist
    IntegerVector pos_idx;
    IntegerVector neg_idx;
    IntegerVector not_neg_idx;
    
//    cout << "passed all declarations" << endl;

//    cout << "starting learningrate" << endl;
//    Rcout << learningrate << endl;
    
    if(any(positive)){
//      cout << "POSITIVES!" << endl;
      arma::uvec idx(sum(positive));
      
      int ci = 0;
      for(unsigned int i=0; i < positive.n_elem; i++){
          if(positive[i] == 1){
               idx[ci] =  i;
               ci++;
          }
      }
      
      // learningrate is a double vector initially filled with 0.1
      NumericVector tmp_pmin = pmin(as<NumericVector>(wrap(learningrate.elem(idx) * as<double>(learningrate_factor["plus"]))), 
                                            Rcpp::rep(as<double>(learningrate_limit["max"]), learningrate.n_elem));
      learningrate.elem(idx) = as<arma::vec>(tmp_pmin);
    }
    
//    cout << "learning after positive" << endl;
//    Rcout << learningrate << endl;
    
    if(any(negative)){
//        cout << "NEGATIVES!" << endl;
        for(unsigned int i=0; i < negative.size(); i++){
          if(negative[i]){
            neg_idx.push_back(i);
          }else{
            not_neg_idx.push_back(i); // the not_neg equivalent
          }
        }
        
        // should be integer so conversion allowed
        //arma::uvec excl = as<arma::uvec>(exclude);
        arma::uvec idx = as<arma::uvec>(neg_idx);
        arma::uvec pidx = as<arma::uvec>(not_neg_idx);
        
//        cout << "set indices" << endl;
        
        // weights list of arma matrices
        // currently doesn't implement exclude values
        // use % for element-wise multiplication with arma vectors
        weights_vec.elem(idx) = weights_vec.elem(idx) +
            gradients_old.elem(idx) % learningrate.elem(idx);
        
        //arma::vec tmp_learn = learningrate.elem(idx);
        
        // use * for element-wise multiplication with NumericVector
        NumericVector tmp_pmax = pmax(as<NumericVector>(wrap(learningrate.elem(idx) *
            as<double>(learningrate_factor["minus"]))), Rcpp::rep(as<double>(learningrate_limit["min"]), learningrate.n_elem));
            
        learningrate.elem(idx) = as<arma::vec>(tmp_pmax);
        gradients_old.elem(idx) = arma::zeros<arma::vec>(idx.n_elem);
        
//        cout << "learningrate after negative" << endl;
//        Rcout << learningrate << endl;
                
        if(pidx.n_elem > 0){
            weights_vec.elem(pidx) = weights_vec(pidx) - 
                sign_gradient.elem(pidx) % learningrate.elem(pidx);
            gradients_old.elem(pidx) = sign_gradient.elem(pidx);
        }
    }else{
//        cout << "other search" << endl;
        // currently missing exclude elements
        weights_vec = weights_vec - sign_gradient % learningrate;
        gradients_old = sign_gradient;
    }
    
//    cout << "plus wieghts" << endl;
//    Rcout << weights_vec << endl;
    
//    cout << "learning rate" << endl;
//    Rcout << learningrate << endl;
    
//    cout << "sign gradient" << endl;
//    Rcout << sign_gradient << endl;
    
    return List::create(Named("gradients_old") = gradients_old, 
                        Named("weights") = c_relist(weights_vec, nrow_weights, ncol_weights),
                        Named("learningrate") = learningrate);
}


inline
List c_backprop(
    arma::vec gradients, 
    List weights, 
    arma::ivec nrow_weights, 
    arma::ivec ncol_weights, 
    double learningrate_bp, 
    SEXP exclude)
{
    //weights <- unlist(weights)
    arma::vec weights_vec = c_unlist(weights);
    
    if (!Rf_isNull(exclude)){
        stop("exclude not yet implemented");
        //weights_vec[-exclude] <- weights_vec[-exclude] - gradients * learningrate_bp
    }else{
        weights_vec = weights_vec - gradients * learningrate_bp;
    } 
//    list(gradients.old = gradients, weights = relist(weights, 
//        nrow.weights, ncol.weights), learningrate = learningrate_bp)
        
    return List::create(Named("gradients_old") = gradients, 
                        Named("weights") = c_relist(weights_vec, nrow_weights, ncol_weights),
                        Named("learningrate") = learningrate_bp);
}


inline
List c_rprop(
  List weights,
  arma::mat response, // big.matrix
  arma::mat covariate, // big.matrix
  double threshold, 
  List learningrate_limit, 
  List learningrate_factor, 
  int stepmax, 
  String lifesign, 
  int lifesign_step, 
  SEXP act_fct, 
  SEXP act_deriv_fct, 
  String act_fct_name,
  SEXP err_fct, 
  SEXP err_deriv_fct, 
  String err_fct_name,
  String algorithm, 
  bool linear_output, 
  SEXP exclude, 
  double learningrate_bp,
  bool dropout,
  double visible_dropout,
  arma::vec hidden_dropout) 
{   
//    cout << "called c_prop" << endl;
  
    // declare various variables
    int step = 1;
    //int nchar_stepmax = max(snprintf(nullptr, 0, "%d", stepmax), 7);
    int length_weights = weights.size();
    int length_unlist = 0;
    bool special;
    int c_exclude;
    
    // declare function pointers
    XPtr<nmfptr2> xptr_err_fct(err_fct);
    XPtr<nmfptr2> xptr_err_deriv_fct(err_deriv_fct);
    XPtr<nmfptr> xptr_act_fct(act_fct);
    XPtr<nmfptr> xptr_act_deriv_fct(act_deriv_fct);    
    XPtr<nmfptr> xptr_output_act_fct = xptr_act_fct;
    XPtr<nmfptr> xptr_output_act_deriv_fct = xptr_act_deriv_fct;
    
//    cout << "declared output xptrs" << endl;
    
    // declare integer vectors
    arma::ivec nrow_weights = arma::zeros<ivec>(length_weights);
    arma::ivec ncol_weights = arma::zeros<ivec>(length_weights);

    for (int i = 0; i < length_weights; i++){
      NumericMatrix tmp = weights[i];
      int tmp_nrow = tmp.nrow();
      int tmp_ncol = tmp.ncol();
      nrow_weights[i] = tmp_nrow;
      ncol_weights[i] = tmp_ncol;
      length_unlist += (tmp_nrow * tmp_ncol);
    }
    
    // default filled with 0's
    arma::vec gradients_old = arma::zeros<arma::vec>(length_unlist);
    arma::vec learningrate = arma::zeros<arma::vec>(length_unlist);
    learningrate.fill(0.1);
    
    if (Rf_isNull( exclude )) {
      c_exclude = length_unlist + 1;
    }
    
    if(act_fct_name == "tanh" || act_fct_name == "logistic" || act_fct_name == "relu"){
      special = true;
    }else{
      special = false;
    }

    if ( linear_output ){
      xptr_output_act_fct = XPtr<nmfptr>(new nmfptr(&output_func_linear));
      xptr_output_act_deriv_fct = XPtr<nmfptr>(new nmfptr(&output_deriv_func_linear));
    }else{
        if( err_fct_name == "ce" && act_fct_name == "logistic" ){
          xptr_err_deriv_fct = err_deriv_func("ce_log", response);
          linear_output = true;
        }
        
        if( err_fct_name == "ce" && act_fct_name == "relu" ){
          xptr_err_deriv_fct = err_deriv_func("ce_log", response);
          linear_output = true;
        }
        
        if( act_fct_name == "relu" ){
            if( response.n_cols < 2){
                //Rcout << "using logistic function for binary problem" << std::endl;
                xptr_output_act_fct = act_func(String("logistic"));
                xptr_output_act_deriv_fct = act_func(String("logistic"));
                //xptr_output_act_fct = XPtr<nmfptr>(new nmfptr(&output_func_linear));
                //xptr_output_act_deriv_fct = XPtr<nmfptr>(new nmfptr(&output_deriv_func_linear));
            }else{
                //multinomial problems use softmax
                stop("softmax not yet implemented for multinomial problems");
            }
        }else{
            //Rcout << "using native activation function" << std::endl;
            xptr_output_act_fct = xptr_act_fct;
            xptr_output_act_deriv_fct = xptr_act_deriv_fct;
        }
    }
 
//    cout << "first compute_net" << endl;
    // compute the neuralnet
    List result = c_compute_net(weights, length_weights, 
                          covariate, // big.matrix
                          xptr_act_fct, 
                          xptr_act_deriv_fct, 
                          xptr_output_act_fct, 
                          xptr_output_act_deriv_fct, 
                          special,
                          dropout,
                          visible_dropout,
                          hidden_dropout);
  
    // make err XPtr functional
    nmfptr2 c_err_deriv_fct = *xptr_err_deriv_fct;
    
//    arma::mat tmp_arma = as<arma::mat>(result["net_result"]);
//    cout << "rprop net.result" << endl;
//    Rcout << tmp_arma.submat(0,0,5,0) << endl;
    
//    cout << "first err_deriv_fct" << endl;
    arma::mat err_deriv = c_err_deriv_fct(as<arma::mat>(result["net_result"]), response);
    
//    cout << "outside err.deriv" << endl;
//    Rcout << err_deriv.submat(0,0,5,0) << endl;

//    cout << "first gradients" << endl;
    arma::vec gradients = c_calculate_gradients(weights,
                                    length_weights, 
                                    result["neurons"], // list of big.matrices
                                    result["neuron_deriv"], // [[1]] is big.matrix
                                    err_deriv, // big.matrix
                                    c_exclude, 
                                    linear_output);
    //cout << "passed c_gradients" << endl;
    
//    cout << "outside gradients" << endl;
//    Rcout << gradients << endl;

//    cout << "the loop starts" << endl;
    double reached_threshold = max(abs(gradients));
    double min_reached_threshold = reached_threshold;
    while (step < stepmax && reached_threshold > threshold) {
        
        if(lifesign != "none" && step%lifesign_step == 0){
            if(step == lifesign_step){
                cout << "\t\t\t\t" << step << "\tmin thresh: " 
                << min_reached_threshold << endl;
            }else{
                cout << right << setw(55) << step << " "
                << "min thresh: " << min_reached_threshold << endl;
            }
        }
        // covert to a switch statement
        if (algorithm == "rprop+") {
          result = c_plus(gradients, gradients_old, weights, 
                         nrow_weights, ncol_weights, learningrate, 
                         learningrate_factor, 
                         learningrate_limit, exclude);
        }else{
          if (algorithm == "backprop") {
//              std::cout << "calling backprop" << std::endl;
              result = c_backprop(gradients, weights,
                               nrow_weights, ncol_weights, learningrate_bp, 
                               exclude);
          }
        }//else{
//            result <- minus(gradients, gradients.old, weights, 
//                            length.weights, nrow.weights, ncol.weights, learningrate, 
//                            learningrate.factor, learningrate.limit, algorithm, 
//                            exclude)
//          } 
//        } 
//        
//        std::cout << "algorithm completed" << std::endl;
        
        gradients_old = as<arma::vec>(result["gradients_old"]);
//        cout << "old gradients" << endl;
//        Rcout << gradients_old << endl;
        
        weights = result["weights"];
        
//        cout << "inside weights" << endl;
//        for(int w = 0; w< weights.size(); w++){
//          Rcout << as<arma::mat>(weights[w]) << endl;
//        }
        
        learningrate = as<arma::vec>(result["learningrate"]);
//        std::cout << "learningrate pulled" << std::endl;
        
        result = c_compute_net(weights, length_weights, 
                              covariate, // big.matrix
                              xptr_act_fct, 
                              xptr_act_deriv_fct, 
                              xptr_output_act_fct, 
                              xptr_output_act_deriv_fct,
                              special,
                              dropout,
                              visible_dropout,
                              hidden_dropout);
                              
//        arma::mat tmp_arma = as<arma::mat>(result["net_result"]);
//        cout << "rprop inside net.result" << endl;
////        Rcout.precision(10);
//        Rcout << tmp_arma.submat(0,0,5,0) << endl;
//    
        err_deriv = c_err_deriv_fct(result["net_result"], response);
//        cout << "inside err.deriv" << endl;
//        Rcout << err_deriv.submat(0,0,5,0) << endl;

//        cout << "neuron_deriv" << endl;
//        List tmp_n_der = result["neuron_deriv"]

        gradients = c_calculate_gradients(weights, 
                                        length_weights, 
                                        result["neurons"], // list of big.matrices 
                                        result["neuron_deriv"], // [[1]] is big.matrix
                                        err_deriv, 
                                        c_exclude, 
                                        linear_output);
          
//        gradients.print("inside gradients");
                                         
        reached_threshold = max(abs(gradients));
//        cout << "reached threshold" << endl;
//        cout << reached_threshold << endl;
        
        if (reached_threshold < min_reached_threshold) {
            min_reached_threshold = reached_threshold;
        }
        step += 1;
    }
//    cout << "\tmin thresh: " << reached_threshold << endl;
    if (lifesign == "full" && reached_threshold > threshold) {
        cout << right << setw(56) << "\tmin thresh: " << min_reached_threshold << endl;
    }
    
    return List::create(Named("weights") = weights,
                        Named("step") = step,
                        Named("reached_threshold") = reached_threshold,
                        Named("net_result") = result["net_result"],
                        Named("neuron_deriv") = result["neuron_deriv"]);
}


inline
double c_round(double val, int p)
{
  double out = double(int(val*std::pow(double(10),p)+.5))/std::pow(double(10),p);
  return out;
}


inline
arma::vec concat_arma_mat(arma::vec a, arma::mat b)
{
  arma::vec b_vec = vectorise(b);
  arma::vec c = join_cols(a, b_vec);
  return c;
}


inline
arma::mat c_calculate_generalized_weights(
  List weights, 
  List neuron_deriv, 
  arma::mat net_result) 
{
//    cout << "called generalized wieghts" << endl;

    // need a deepcopy to make sure changes don't propagate up
    List weights_copy = clone(weights);
    
    for (int w=0; w < weights_copy.size(); w++) {
        weights_copy[w] = remove_intercept(as<arma::mat>(weights_copy[w]));
    }
    arma::mat generalized_weights;
    arma::mat delta;
    
//    cout << "in the loop" << endl;
    for (unsigned int k=0; k < net_result.n_cols; k++){
        for (int w=(weights_copy.size()-1); w > -1; w--){
            if ( w == (weights_copy.size()-1)){
//              cout << "first iteration" << endl;
                arma::mat temp = as<arma::mat>(neuron_deriv[w]).col(k) %
                    (1/(net_result.col(k) % (1 - net_result.col(k))));
                  
                delta = temp * trans(as<arma::mat>(weights_copy[w]).col(k));
            }else{
//              cout << "last iterations" << endl;
              delta = (delta % as<arma::mat>(neuron_deriv[w])) * 
                trans(as<arma::mat>(weights_copy[w]));
            }
        }
        generalized_weights.insert_cols(generalized_weights.n_cols, delta);
    }
    return generalized_weights;
}

#endif
