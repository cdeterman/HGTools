#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>    // std::isnan
#include <string>   // std::string
#include <cstdio>   // snprintf
#include <iomanip>  // setprecision

#include "neuralnet_functions.hpp"

// With RcppArmadillo you don't include Rcpp.h
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, BH, bigmemory)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

// provide definitions for BigMatrix objects
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

// Enable C++11 via this plugin
// [[Rcpp::plugins(cpp11)]]

typedef vector<std::string> Names;

// @export
//[[Rcpp::export]]
List c_generate_initial_variables_bm(
  SEXP data,
  SEXP covariate_bm,
  SEXP response_bm,
  List model_list,
  String act_fct,
  String err_fct){
    
    XPtr<BigMatrix> xpData(data);
    arma::mat data_arma((double *)xpData->matrix(), 
                            xpData->nrow(), 
                            xpData->ncol(), 
                            false);
                          
    // make covariate & response BMs accessible
    XPtr<BigMatrix> xpCovariateBM(covariate_bm);
    XPtr<BigMatrix> xpResponseBM(response_bm);
    MatrixAccessor<double> A_CovariateBM(*xpCovariateBM);
    MatrixAccessor<double> A_ResponseBM(*xpResponseBM);
    
    Names response = as<Names>(model_list["response"]);
    Names variables = as<Names>(model_list["variables"]);
    int num_ivs = response.size();
    int num_dvs = variables.size();
    int nrows = data_arma.n_rows;
    
//    Names colnames = data.names();
    Names colnames = xpData->column_names();
    
    arma::mat intercept = ones<arma::mat>(data_arma.n_rows, 1);
    arma::uvec idxr(num_ivs);
    arma::uvec idxv(num_dvs);
    
    arma::mat c_covariate(nrows, num_dvs+1);
    arma::mat c_response(nrows, num_ivs);
    arma::mat dvs(nrows, num_dvs);
    
    //cout << "initialized all objects" << endl;
    
    // find the column indices
    for (int i=0; i < num_ivs; i++){
      idxr[i] = distance(colnames.begin(), find(colnames.begin(), colnames.end(), response[i]));
    }
    
    for (int i=0; i < num_dvs; i++){
      idxv[i] = distance(colnames.begin(), find(colnames.begin(), colnames.end(), variables[i]));
    }
    
    // subset the columns to be used for the model
    arma::mat sub_covariate = data_arma.cols(idxv);
    //cout << "idxv subset complete" << endl;
    c_covariate = join_rows(intercept, sub_covariate);
    //cout << "covariate successfully created" << endl;
    c_response = data_arma.cols(idxr);
    
    // convert covariate output to big.matrix
    int nr = c_covariate.n_rows;
    int nc = c_covariate.n_cols;
    
    for(int i =0; i < nc; i++){
      for(int j=0; j<nr; j++){
        A_CovariateBM[i][j] = c_covariate(j,i);
      }
    }
    
    // convert response output to big.matrix
    nr = c_response.n_rows;
    nc = c_response.n_cols;
    for(int i =0; i < nc; i++){
      for(int j=0; j<nr; j++){
        A_ResponseBM[i][j] = c_response(j,i);
      }
    }
    
    // functions
    XPtr<nmfptr> xp_act_fct = act_func(act_fct);
    XPtr<nmfptr> xp_act_deriv_fct = act_deriv_func(act_fct);
    XPtr<nmfptr2> xp_err_fct = err_func(err_fct, c_response);
    XPtr<nmfptr2> xp_err_deriv_fct = err_deriv_func(err_fct, c_response);    
    
    return List::create(Named("err.fct") = xp_err_fct, 
                        Named("err.deriv.fct") = xp_err_deriv_fct,
                        Named("act.fct") = xp_act_fct,
                        Named("act.deriv.fct") = xp_act_deriv_fct
                        );
  }  


//[[Rcpp::export]]
List c_calculate_neuralnet_bm(
  SEXP data,
  List model_list, 
  IntegerVector hidden, 
  SEXP stepmax, 
  SEXP rep, 
  SEXP threshold, 
  List learningrate_limit, 
  List learningrate_factor, 
  const String lifesign, 
  SEXP covariate, // BigMatrix
  SEXP response,  // BigMatrix
  SEXP lifesign_step, 
  SEXP startweights, 
  const String algorithm, 
  SEXP act_fct, 
  SEXP act_deriv_fct, 
  String act_fct_name,
  SEXP err_fct, 
  SEXP err_deriv_fct, 
  String err_fct_name,
  SEXP linear_output, 
  SEXP likelihood, 
  SEXP exclude, 
  SEXP constant_weights, 
  SEXP learningrate_bp) 
{
//    cout << "called calculate_neuralnet" << endl;
    
    // convert SEXP objects to C types
    int c_stepmax = as<int>(stepmax);
    int c_rep = as<int>(rep);
    int c_lifesign_step = as<int>(lifesign_step);
    double c_threshold = as<double>(threshold);
    bool c_linear_output = as<bool>(linear_output);
    bool c_likelihood = as<bool>(likelihood);
    
    // Deal with BigMatrix objects
    XPtr<BigMatrix> xpResponse(response);
    XPtr<BigMatrix> xpCovariate(covariate);
    arma::mat response_arma((double *)xpResponse->matrix(), 
                            xpResponse->nrow(), 
                            xpResponse->ncol(), 
                            false);
    arma::mat covariate_arma((double *)xpCovariate->matrix(), 
                            xpCovariate->nrow(), 
                            xpCovariate->ncol(), 
                            false);
    
    // convert response to arma::mat
    // first convert to NumericMatrix
//    NumericMatrix nm_response = DFtoNM(response);
//    // now convert to arma::mat
//    int n = nm_response.nrow(), k = nm_response.ncol();
//    // create armadillo matrix, reuse memory
//    arma::mat response_arma(nm_response.begin(), n, k, false);
    
    // Declare other variables
    double aic;
    double bic;
    
//    cout << "everything initialized" << endl;
    
    if(!(Rf_isNull(startweights))){
        stop("custom startweights not yet implemented");
//        cout << "custom startweights not yet implemented" << endl;
//        return 0;
    }
    
    // get starting time may just move to front function
    //time_t time_start = time(nullptr);
    //time_start_local <- localtime(&result);
    //const clock_t begin_time = clock();
    
    XPtr<nmfptr2> xptr_err_fct = err_func(err_fct_name, response_arma);
    nmfptr2 c_err_fct = *xptr_err_fct;
    
//    cout << "Initialized xptr" << endl;
    
    // generate initial start weights
    List result = c_generate_startweights(model_list, hidden, startweights, 
                                          c_rep, exclude, constant_weights);
//    cout << "passed generate_startweights" << endl;
                                          
    List weights = result["weights"];
    
    // not pulling exclude because it isn't modified
    //exclude <- result$exclude
    
    // declare integer vectors
    int length_weights = weights.size();
    arma::ivec nrow_weights = zeros<ivec>(length_weights);
    arma::ivec ncol_weights = zeros<ivec>(length_weights);

    for (int i = 0; i < length_weights; i++){
      NumericMatrix tmp = weights[i];
      int tmp_nrow = tmp.nrow();
      int tmp_ncol = tmp.ncol();
      nrow_weights[i] = tmp_nrow;
      ncol_weights[i] = tmp_ncol;
    }

//    cout << "just before c_prop" << endl;

    result = c_rprop(weights,  
                    response_arma, // big.matrix
                    covariate_arma, // big.matrix
                    c_threshold,
                    learningrate_limit, 
                    learningrate_factor, 
                    c_stepmax, 
                    lifesign, c_lifesign_step, 
                    act_fct, act_deriv_fct, act_fct_name, 
                    err_fct, err_deriv_fct, err_fct_name,
                    algorithm, c_linear_output, 
                    exclude, learningrate_bp);
                    
//    cout << "finished r_prop" << endl;
                    
    List dx_startweights = weights;
    weights = result["weights"];
    int step = as<int>(result["step"]);
    double reached_threshold = as<double>(result["reached_threshold"]);
    
    arma::mat net_result = as<arma::mat>(result["net_result"]);
    
    double error = sum(sum(c_err_fct(net_result, response_arma)));
//    cout << "error calculation successful" << endl;
//    cout << error << endl;
    
    if (std::isnan(error) & err_fct_name == "ce") {
        stop("error is na, method to remove na values in progress");
//        cout << "error is na, method to remove na values in progress" << endl;
//        return 0;
//      if (all(net_result <= 1, net_result >= 0)) {
//        // need method to remove na values (e.g. na.rm=T)
//        error = sum(sum(c_err_fct(net_result, response)))       
//      }
    }
        
//    if (!is.null(constant_weights) && any(constant_weights != 0)) {
//      exclude <- exclude[-which(constant_weights != 0)]
//    }

//    if (length(exclude) == 0) {exclude <- NULL}
//    aic <- NULL
//    bic <- NULL
    if (c_likelihood) {
        
        arma::vec weights_vec = c_unlist(weights);
        
        // normally has the -length(exclude) but not yet implemented
        int synapse_count = weights_vec.n_elem; // -length(exclude)
        aic = 2 * error + (2 * synapse_count);
        bic = 2 * error + log(response_arma.n_rows) * synapse_count;
    }
    
//    cout << "likelihood calculations successful" << endl;

    if (std::isnan(error)) {
        stop("'err_fct' does not fit 'data' or 'act_fct'");
//      cout << "'err_fct' does not fit 'data' or 'act_fct'" << endl;
//      return 0;
    }
    
    // If user wants verbose output that processes are running
    if (lifesign != "none") {
        if (reached_threshold <= c_threshold) {
            // print spaces
            string spaces = string(max(snprintf(nullptr, 0, "%d", c_stepmax), 7) - 
                snprintf(nullptr, 0, "%d", step),
                ' ');
            cout << string(55 - snprintf(nullptr, 0, "%d", step), ' ') << step;
            
            // print aic
            spaces = string(snprintf(nullptr, 0, "%f", c_round(error, 5)) - 
                snprintf(nullptr, 0, "%f", c_round(error, 0)),
                ' ');
            cout << "\terror: " << c_round(error, 5) << spaces;
            if(aic){
              spaces = string(snprintf(nullptr, 0, "%f", c_round(aic, 5)) - 
                snprintf(nullptr, 0, "%f", c_round(aic, 0)),
                ' ');
              cout << "\taic: " << c_round(aic, 5) << spaces;
            }
            
            // print bic
            if(bic){
              spaces = string(snprintf(nullptr, 0, "%f", c_round(bic, 5)) - 
                snprintf(nullptr, 0, "%f", c_round(bic, 0)),
                ' ');
              cout << "\tbic: " << c_round(bic, 5) << spaces;
            }
        }
    }
 
    if ( reached_threshold > c_threshold ) {
      return List::create(Named("output.vector") = R_NilValue,
                          Named("weights") = R_NilValue);
    }
//    else{
//      cout << "threshold low enough!" << endl;
//    }

//    cout << "passed threshold condition" << endl;
    
    //NumericVector output_vector;
    arma::vec output_vector(5);
    
    if(!c_likelihood) {
//      output_vector = NumericVector::create(
//        _["error"] = error, 
//        _["reached_threshold"] = reached_threshold, 
//        _["step"] = step);

        // C++11 initializer
//        output_vector = {error, reached_threshold, double(step)};
        output_vector(0) = error;
        output_vector(1) = reached_threshold;
        output_vector(2) = double(step);
        output_vector.resize(3);
    }else{
//        output_vector = NumericVector::create(
//          _["error"] = error, 
//          _["reached_threshold"] = reached_threshold, 
//          _["step"] = step,
//          _["aic"] = aic,
//          _["bic"] = bic);
//        // C++11 initializer
//        output_vector = {error, reached_threshold, double(step), aic, bic};
        output_vector(0) = error;
        output_vector(1) = reached_threshold;
        output_vector(2) = double(step);
        output_vector(3) = aic;
        output_vector(4) = bic;
    }

    for (int w=0; w < weights.size(); w++){
        output_vector = concat_arma_mat(output_vector, weights[w]);
    }
    
    arma::mat generalized_weights;
    generalized_weights = c_calculate_generalized_weights(weights, result["neuron_deriv"], net_result);
        
    arma::vec startweights_vec = c_unlist(dx_startweights);
    arma::vec weights_vec = c_unlist(weights);
    
    if (!(Rf_isNull(exclude))) {
        stop("exclude option not yet implemented");
//        cout << "exclude option not yet implemented" << endl;
//        return 0;
        // Primary problem, cannot return NA in a double vector
        
        //startweights_vec[exclude] = R_NilValue;
        //weights_vec[exclude] = R_NilValue;
    }
    
    List startweights_list = c_relist(startweights_vec, nrow_weights, ncol_weights);
    weights = c_relist(weights_vec, nrow_weights, ncol_weights);
    
    // convert all weights to SEXP objects
    for (int i=0; i < weights.size(); i++){
        weights[i] = wrap(weights[i]);
    }
    // convert all start_weights to SEXP objects
    for (int i=0; i < startweights_list.size(); i++){
        startweights_list[i] = wrap(startweights_list[i]);
    }
    
//    cout << "converted lists to SEXP" << endl;
    
    NumericVector output_vector_nv = as<NumericVector>(wrap(output_vector));
      
    return List::create(Named("generalized.weights") = wrap(generalized_weights),
                        Named("weights") = weights,
                        Named("startweights") = startweights_list,
                        Named("net.result") = wrap(net_result),
                        Named("output.vector") = output_vector_nv);
}