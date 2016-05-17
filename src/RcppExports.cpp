// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// c_compute
List c_compute(List nn, NumericMatrix covariate_in, bool dropout, double visible_dropout, arma::vec hidden_dropout);
RcppExport SEXP HGTools_c_compute(SEXP nnSEXP, SEXP covariate_inSEXP, SEXP dropoutSEXP, SEXP visible_dropoutSEXP, SEXP hidden_dropoutSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< List >::type nn(nnSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type covariate_in(covariate_inSEXP);
    Rcpp::traits::input_parameter< bool >::type dropout(dropoutSEXP);
    Rcpp::traits::input_parameter< double >::type visible_dropout(visible_dropoutSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hidden_dropout(hidden_dropoutSEXP);
    __result = Rcpp::wrap(c_compute(nn, covariate_in, dropout, visible_dropout, hidden_dropout));
    return __result;
END_RCPP
}
// c_compute_bm
List c_compute_bm(List nn, SEXP covariate_in, bool dropout, double visible_dropout, arma::vec hidden_dropout);
RcppExport SEXP HGTools_c_compute_bm(SEXP nnSEXP, SEXP covariate_inSEXP, SEXP dropoutSEXP, SEXP visible_dropoutSEXP, SEXP hidden_dropoutSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< List >::type nn(nnSEXP);
    Rcpp::traits::input_parameter< SEXP >::type covariate_in(covariate_inSEXP);
    Rcpp::traits::input_parameter< bool >::type dropout(dropoutSEXP);
    Rcpp::traits::input_parameter< double >::type visible_dropout(visible_dropoutSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hidden_dropout(hidden_dropoutSEXP);
    __result = Rcpp::wrap(c_compute_bm(nn, covariate_in, dropout, visible_dropout, hidden_dropout));
    return __result;
END_RCPP
}
// c_generate_initial_variables
List c_generate_initial_variables(DataFrame data, List model_list, String act_fct, String err_fct);
RcppExport SEXP HGTools_c_generate_initial_variables(SEXP dataSEXP, SEXP model_listSEXP, SEXP act_fctSEXP, SEXP err_fctSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< List >::type model_list(model_listSEXP);
    Rcpp::traits::input_parameter< String >::type act_fct(act_fctSEXP);
    Rcpp::traits::input_parameter< String >::type err_fct(err_fctSEXP);
    __result = Rcpp::wrap(c_generate_initial_variables(data, model_list, act_fct, err_fct));
    return __result;
END_RCPP
}
// c_calculate_neuralnet
List c_calculate_neuralnet(DataFrame data, List model_list, IntegerVector hidden, SEXP stepmax, SEXP rep, SEXP threshold, List learningrate_limit, List learningrate_factor, const String lifesign, DataFrame covariate, DataFrame response, SEXP lifesign_step, SEXP startweights, const String algorithm, SEXP act_fct, SEXP act_deriv_fct, String act_fct_name, SEXP err_fct, SEXP err_deriv_fct, String err_fct_name, SEXP linear_output, SEXP likelihood, SEXP exclude, SEXP constant_weights, SEXP learningrate_bp, SEXP dropout, SEXP visible_dropout, SEXP hidden_dropout);
RcppExport SEXP HGTools_c_calculate_neuralnet(SEXP dataSEXP, SEXP model_listSEXP, SEXP hiddenSEXP, SEXP stepmaxSEXP, SEXP repSEXP, SEXP thresholdSEXP, SEXP learningrate_limitSEXP, SEXP learningrate_factorSEXP, SEXP lifesignSEXP, SEXP covariateSEXP, SEXP responseSEXP, SEXP lifesign_stepSEXP, SEXP startweightsSEXP, SEXP algorithmSEXP, SEXP act_fctSEXP, SEXP act_deriv_fctSEXP, SEXP act_fct_nameSEXP, SEXP err_fctSEXP, SEXP err_deriv_fctSEXP, SEXP err_fct_nameSEXP, SEXP linear_outputSEXP, SEXP likelihoodSEXP, SEXP excludeSEXP, SEXP constant_weightsSEXP, SEXP learningrate_bpSEXP, SEXP dropoutSEXP, SEXP visible_dropoutSEXP, SEXP hidden_dropoutSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< List >::type model_list(model_listSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type hidden(hiddenSEXP);
    Rcpp::traits::input_parameter< SEXP >::type stepmax(stepmaxSEXP);
    Rcpp::traits::input_parameter< SEXP >::type rep(repSEXP);
    Rcpp::traits::input_parameter< SEXP >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< List >::type learningrate_limit(learningrate_limitSEXP);
    Rcpp::traits::input_parameter< List >::type learningrate_factor(learningrate_factorSEXP);
    Rcpp::traits::input_parameter< const String >::type lifesign(lifesignSEXP);
    Rcpp::traits::input_parameter< DataFrame >::type covariate(covariateSEXP);
    Rcpp::traits::input_parameter< DataFrame >::type response(responseSEXP);
    Rcpp::traits::input_parameter< SEXP >::type lifesign_step(lifesign_stepSEXP);
    Rcpp::traits::input_parameter< SEXP >::type startweights(startweightsSEXP);
    Rcpp::traits::input_parameter< const String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< SEXP >::type act_fct(act_fctSEXP);
    Rcpp::traits::input_parameter< SEXP >::type act_deriv_fct(act_deriv_fctSEXP);
    Rcpp::traits::input_parameter< String >::type act_fct_name(act_fct_nameSEXP);
    Rcpp::traits::input_parameter< SEXP >::type err_fct(err_fctSEXP);
    Rcpp::traits::input_parameter< SEXP >::type err_deriv_fct(err_deriv_fctSEXP);
    Rcpp::traits::input_parameter< String >::type err_fct_name(err_fct_nameSEXP);
    Rcpp::traits::input_parameter< SEXP >::type linear_output(linear_outputSEXP);
    Rcpp::traits::input_parameter< SEXP >::type likelihood(likelihoodSEXP);
    Rcpp::traits::input_parameter< SEXP >::type exclude(excludeSEXP);
    Rcpp::traits::input_parameter< SEXP >::type constant_weights(constant_weightsSEXP);
    Rcpp::traits::input_parameter< SEXP >::type learningrate_bp(learningrate_bpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type dropout(dropoutSEXP);
    Rcpp::traits::input_parameter< SEXP >::type visible_dropout(visible_dropoutSEXP);
    Rcpp::traits::input_parameter< SEXP >::type hidden_dropout(hidden_dropoutSEXP);
    __result = Rcpp::wrap(c_calculate_neuralnet(data, model_list, hidden, stepmax, rep, threshold, learningrate_limit, learningrate_factor, lifesign, covariate, response, lifesign_step, startweights, algorithm, act_fct, act_deriv_fct, act_fct_name, err_fct, err_deriv_fct, err_fct_name, linear_output, likelihood, exclude, constant_weights, learningrate_bp, dropout, visible_dropout, hidden_dropout));
    return __result;
END_RCPP
}
// c_generate_initial_variables_bm
List c_generate_initial_variables_bm(SEXP response_bm, List model_list, String act_fct, String err_fct);
RcppExport SEXP HGTools_c_generate_initial_variables_bm(SEXP response_bmSEXP, SEXP model_listSEXP, SEXP act_fctSEXP, SEXP err_fctSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type response_bm(response_bmSEXP);
    Rcpp::traits::input_parameter< List >::type model_list(model_listSEXP);
    Rcpp::traits::input_parameter< String >::type act_fct(act_fctSEXP);
    Rcpp::traits::input_parameter< String >::type err_fct(err_fctSEXP);
    __result = Rcpp::wrap(c_generate_initial_variables_bm(response_bm, model_list, act_fct, err_fct));
    return __result;
END_RCPP
}
// c_calculate_neuralnet_bm
List c_calculate_neuralnet_bm(SEXP data, List model_list, IntegerVector hidden, int stepmax, int rep, double threshold, List learningrate_limit, List learningrate_factor, const String lifesign, SEXP covariate, SEXP response, int lifesign_step, SEXP startweights, const String algorithm, SEXP act_fct, SEXP act_deriv_fct, String act_fct_name, SEXP err_fct, SEXP err_deriv_fct, String err_fct_name, bool linear_output, bool likelihood, SEXP exclude, SEXP constant_weights, SEXP learningrate_bp, bool dropout, double visible_dropout, arma::vec hidden_dropout);
RcppExport SEXP HGTools_c_calculate_neuralnet_bm(SEXP dataSEXP, SEXP model_listSEXP, SEXP hiddenSEXP, SEXP stepmaxSEXP, SEXP repSEXP, SEXP thresholdSEXP, SEXP learningrate_limitSEXP, SEXP learningrate_factorSEXP, SEXP lifesignSEXP, SEXP covariateSEXP, SEXP responseSEXP, SEXP lifesign_stepSEXP, SEXP startweightsSEXP, SEXP algorithmSEXP, SEXP act_fctSEXP, SEXP act_deriv_fctSEXP, SEXP act_fct_nameSEXP, SEXP err_fctSEXP, SEXP err_deriv_fctSEXP, SEXP err_fct_nameSEXP, SEXP linear_outputSEXP, SEXP likelihoodSEXP, SEXP excludeSEXP, SEXP constant_weightsSEXP, SEXP learningrate_bpSEXP, SEXP dropoutSEXP, SEXP visible_dropoutSEXP, SEXP hidden_dropoutSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type data(dataSEXP);
    Rcpp::traits::input_parameter< List >::type model_list(model_listSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type hidden(hiddenSEXP);
    Rcpp::traits::input_parameter< int >::type stepmax(stepmaxSEXP);
    Rcpp::traits::input_parameter< int >::type rep(repSEXP);
    Rcpp::traits::input_parameter< double >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< List >::type learningrate_limit(learningrate_limitSEXP);
    Rcpp::traits::input_parameter< List >::type learningrate_factor(learningrate_factorSEXP);
    Rcpp::traits::input_parameter< const String >::type lifesign(lifesignSEXP);
    Rcpp::traits::input_parameter< SEXP >::type covariate(covariateSEXP);
    Rcpp::traits::input_parameter< SEXP >::type response(responseSEXP);
    Rcpp::traits::input_parameter< int >::type lifesign_step(lifesign_stepSEXP);
    Rcpp::traits::input_parameter< SEXP >::type startweights(startweightsSEXP);
    Rcpp::traits::input_parameter< const String >::type algorithm(algorithmSEXP);
    Rcpp::traits::input_parameter< SEXP >::type act_fct(act_fctSEXP);
    Rcpp::traits::input_parameter< SEXP >::type act_deriv_fct(act_deriv_fctSEXP);
    Rcpp::traits::input_parameter< String >::type act_fct_name(act_fct_nameSEXP);
    Rcpp::traits::input_parameter< SEXP >::type err_fct(err_fctSEXP);
    Rcpp::traits::input_parameter< SEXP >::type err_deriv_fct(err_deriv_fctSEXP);
    Rcpp::traits::input_parameter< String >::type err_fct_name(err_fct_nameSEXP);
    Rcpp::traits::input_parameter< bool >::type linear_output(linear_outputSEXP);
    Rcpp::traits::input_parameter< bool >::type likelihood(likelihoodSEXP);
    Rcpp::traits::input_parameter< SEXP >::type exclude(excludeSEXP);
    Rcpp::traits::input_parameter< SEXP >::type constant_weights(constant_weightsSEXP);
    Rcpp::traits::input_parameter< SEXP >::type learningrate_bp(learningrate_bpSEXP);
    Rcpp::traits::input_parameter< bool >::type dropout(dropoutSEXP);
    Rcpp::traits::input_parameter< double >::type visible_dropout(visible_dropoutSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hidden_dropout(hidden_dropoutSEXP);
    __result = Rcpp::wrap(c_calculate_neuralnet_bm(data, model_list, hidden, stepmax, rep, threshold, learningrate_limit, learningrate_factor, lifesign, covariate, response, lifesign_step, startweights, algorithm, act_fct, act_deriv_fct, act_fct_name, err_fct, err_deriv_fct, err_fct_name, linear_output, likelihood, exclude, constant_weights, learningrate_bp, dropout, visible_dropout, hidden_dropout));
    return __result;
END_RCPP
}
