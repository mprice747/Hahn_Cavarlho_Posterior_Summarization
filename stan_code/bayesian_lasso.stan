
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  vector[N] y;      // outcome vector
  real<lower=0> a_tau; // prior precision alpha
  real<lower=0> b_tau; // prior precision beta
  real<lower=0> a_lambda; // prior lambda^2 alpha
  real<lower=0> b_lambda; // prior lambda^2 beta 
}


parameters {
  vector[K] beta;       // coefficients for predictors
  real<lower=0> precision; // precision
  real<lower=0> lambda_sq; // Hyperprior for Lambda 
}


model {
  // Sample lambda_sq and precision
  lambda_sq ~ gamma(a_lambda, b_lambda);
  precision ~ gamma(a_tau, b_tau);
  
  // Sample from double exponential 
  for (k in 1:K)
    beta[k] ~ double_exponential(0, (1/sqrt(precision)) * 1/sqrt(lambda_sq));
    
  // Likelihood 
  y ~ normal(X* beta, sqrt(1/precision));  
}

