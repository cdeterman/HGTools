#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>    // std::isnan
#include <string>   // std::string
#include <cstdio>   // snprintf
#include <iomanip>  // setprecision

#include "neuralnet_functions.hpp"

// With RcppArmadillo you don't include Rcpp.h
//#include <Rcpp.h>
#include <RcppArmadillo.h>

// Enable C++11 via this plugin
// [[Rcpp::plugins(cpp11)]]

// include armadillo
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

// This function will require a preproccessing to make sure the column names in the formula
// do exist in the data.frame/matrix.
// @export
//[[Rcpp::export]]
List c_generate_initial_variables(
  DataFrame data,
  List model_list,
  String act_fct,
  String err_fct){
    
    // convert DataFrame to arma::mat
    // first convert to NumericMatrix
    NumericMatrix nm_data = DFtoNM(data);
    // now convert to arma::mat
    int n = nm_data.nrow(), k = nm_data.ncol();
    // create armadillo matrix, reuse memory
    arma::mat data_arma(nm_data.begin(), n, k, false);
    
    vector< string > response = as<vector< string > >(model_list["response"]);
    vector< string > variables = as<vector< string > >(model_list["variables"]);
    int num_ivs = response.size();
    int num_dvs = variables.size();
    int nrows = data_arma.n_rows;
    
    vector< string > colnames = data.names();
    
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
    
    // functions
    XPtr<nmfptr> xp_act_fct = act_func(act_fct);
    XPtr<nmfptr> xp_act_deriv_fct = act_deriv_func(act_fct);
    XPtr<nmfptr2> xp_err_fct = err_func(err_fct, c_response);
    XPtr<nmfptr2> xp_err_deriv_fct = err_deriv_func(err_fct, c_response);    
    
    return List::create(Named("covariate") = wrap(c_covariate), 
                        Named("response") = wrap(c_response),
                        Named("err.fct") = xp_err_fct, 
                        Named("err.deriv.fct") = xp_err_deriv_fct,
                        Named("act.fct") = xp_act_fct,
                        Named("act.deriv.fct") = xp_act_deriv_fct
                        );
  }  
  

// @export
//[[Rcpp::export]]
List c_calculate_neuralnet(
  DataFrame data,
  List model_list, 
  IntegerVector hidden, 
  SEXP stepmax, 
  SEXP rep, 
  SEXP threshold, 
  List learningrate_limit, 
  List learningrate_factor, 
  const String lifesign, 
  DataFrame covariate, // possibly have converted to matrix first
  DataFrame response,  // possibly have converted to matrix first
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
  SEXP learningrate_bp,
  SEXP dropout,
  SEXP visible_dropout,
  SEXP hidden_dropout) 
{
//    cout << "called calculate_neuralnet" << endl;
    
    // convert SEXP objects to C types
    int c_stepmax = as<int>(stepmax);
    int c_rep = as<int>(rep);
    int c_lifesign_step = as<int>(lifesign_step);
    double c_threshold = as<double>(threshold);
    bool c_dropout = as<bool>(dropout);
    double c_visible_dropout;
    arma::vec c_hidden_dropout;
    
    double c_learningrate_bp;
    if(!Rf_isNull(learningrate_bp)){
        c_learningrate_bp = as<double>(learningrate_bp);
    }
    
    if(c_dropout){
        c_visible_dropout = as<double>(visible_dropout);
        c_hidden_dropout = as<arma::vec>(hidden_dropout);
    }
        
//    cout << "initialized numbers" << endl;
    
//    const String c_lifesign(lifesign);
//    const String c_algorithm(algorithm);
    
    //String act_fct_name = act_fct;  // need the name of the method for later
    //String err_fct_name = err_fct;  // need the name of the method for later
    bool c_linear_output = as<bool>(linear_output);
    bool c_likelihood = as<bool>(likelihood);
    
    // convert response to arma::mat
    // first convert to NumericMatrix
    NumericMatrix nm_response = DFtoNM(response);
    // now convert to arma::mat
    int n = nm_response.nrow(), k = nm_response.ncol();
    // create armadillo matrix, reuse memory
    arma::mat response_arma(nm_response.begin(), n, k, false);
    
    // convert covariate to arma
    // first convert to NumericMatrix
    NumericMatrix nm_covariate = DFtoNM(covariate);
    // now convert to arma::mat
    n = nm_covariate.nrow(), k = nm_covariate.ncol();
    // create armadillo matrix, reuse memory
    arma::mat covariate_arma(nm_covariate.begin(), n, k, false);
    
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
    
//    XPtr<nmfptr2> xptr_err_fct = err_func(err_fct_name, response_arma);
//    nmfptr2 c_err_fct = *xptr_err_fct;

    XPtr<nmfptr2> xptr_err_fct(err_fct);
    nmfptr2 c_err_fct = *xptr_err_fct;
    
//    std::cout << "check relu ce" << std::endl;
    
//    if( act_fct_name == "relu" && err_fct_name == "ce" ){
//        Rcout << "relu ce log" << std::endl;
//        XPtr<nmfptr2> xptr_err_ce_log_fct = err_func("ce_log", response_arma);
//        c_err_fct = *xptr_err_ce_log_fct;
//    }
    
//    cout << "Initialized xptr" << endl;
    
    // generate initial start weights
    List result = c_generate_startweights(model_list, hidden, startweights, 
                                          c_rep, exclude, constant_weights);
//    cout << "passed generate_startweights" << endl;
                                          
    List weights = result["weights"];
    
//    cout << "first weights" << endl;
//    for(int i=0; i < weights.size(); i++){
//      arma::mat tmp = weights[i];
//      cout << tmp << endl;
//      //cout << tmp.n_rows << endl;
//    }
    
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
                    exclude, c_learningrate_bp,
                    c_dropout, c_visible_dropout,
                    c_hidden_dropout);
                    
//    cout << "finished r_prop" << endl;
                    
    List dx_startweights = weights;
    weights = result["weights"];
    int step = as<int>(result["step"]);
    double reached_threshold = as<double>(result["reached_threshold"]);
    
    //cout << reached_threshold << endl;
    
    arma::mat net_result = as<arma::mat>(result["net_result"]);

//    cout << "just before error calculation" << endl;

//    cout << "weights" << endl;
//    for(int w = 0; w<weights.size(); w++){
//      Rcout << as<arma::mat>(weights[w]) << endl;
//    }
//    
//    Rcout << net_result << std::endl;
//    Rcout << net_result.submat(0,0,5,0) << endl;
//    
//    Rcout << "response" << endl;
//    Rcout << response_arma.submat(0,0,5,0);
//
//    Rcout << "err.fct output" << endl;
//    arma::mat tmp_out = c_err_fct(net_result, response_arma);
//    Rcout << tmp_out << std::endl;
//    Rcout << tmp_out.submat(0,0,5,0)  << endl;

//    cout << "diffs" << endl;
//    cout << net_result - response_arma << std::endl;
    
    double error = sum(sum(c_err_fct(net_result, response_arma)));
//    cout << "error calculation successful" << endl;
//    cout << error << endl;
    
    if (std::isnan(error) & (err_fct_name == "ce")) {
        //stop("error is na, method to remove na values in progress");
        
        arma::mat tmp_errs = c_err_fct(net_result, response_arma);
        // change non-finite elements to zero
        tmp_errs.elem( find_nonfinite(tmp_errs) ).zeros();
        
        // re-sum error values
        error = sum(sum(tmp_errs));
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
    
//    cout << "aic/bic addition successful" << endl;
//    for(int i =0; i < weights.size(); i++){
//        Rcout << as<arma::mat>(weights[i]) << endl;
//    }

    for (int w=0; w < weights.size(); w++){
        output_vector = concat_arma_mat(output_vector, weights[w]);
    }
    
//    cout << "after concat_arma_mat" << endl;
//    for(int i =0; i < weights.size(); i++){
//        Rcout << as<arma::mat>(weights[i]) << endl;
//    }
    
//    cout << "added weights successfully" << endl;
//    cout << "weights copy" << endl;
//    for(int i =0; i < weights_copy.size(); i++){
//        Rcout << as<arma::mat>(weights_copy[i]) << endl;
//    }
    
    arma::mat generalized_weights;
    generalized_weights = c_calculate_generalized_weights(weights, result["neuron_deriv"], net_result);
        
//    cout << "passed generalized weights" << endl;
//    for(int i =0; i < weights.size(); i++){
//        Rcout << as<arma::mat>(weights[i]) << endl;
//    }
//    cout << "weights copy" << endl;
//    for(int i =0; i < weights_copy.size(); i++){
//        Rcout << as<arma::mat>(weights_copy[i]) << endl;
//    }
        
    arma::vec startweights_vec = c_unlist(dx_startweights);
    
//    cout << "after startweights unlist" << endl;
//    for(int i =0; i < weights.size(); i++){
//        Rcout << as<arma::mat>(weights[i]) << endl;
//    }
    arma::vec weights_vec = c_unlist(weights);
    
//    cout << "finished unlisting" << endl;
    
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
    
//    cout << "weights finished" << endl;
    
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
//    output_vector_nv.attr("dimnames") = List::create(
//      CharacterVector::create("error","reach_threshold","steps","aic","bic", 
//                              Rcpp::rep("", output_vector_nv.size() - 5)),
//      NumericVector::create(1));
      
    return List::create(Named("generalized.weights") = wrap(generalized_weights),
                        Named("weights") = weights,
                        Named("startweights") = startweights_list,
                        Named("net.result") = wrap(net_result),
                        Named("output.vector") = output_vector_nv);
                        }
                        
