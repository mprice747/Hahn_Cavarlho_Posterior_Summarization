
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  vector[N] y;      // outcome vector
  vector[K] A;  // scale for half cauchy
  real<lower=0> a_tau; // prior precision alpha
  real<lower=0> b_tau; // prior precision beta
  vector[K] c; 
}
parameters {
  vector[K] beta;       // coefficients for predictors
  real<lower=0> precision; // error scale
  vector<lower=0>[K] lambda; // variance for betas
}

model {
  
  // sample lambda
  for (k in 1:K)
    lambda[k] ~ cauchy(0, A[k]);
  
  // sample beta
  for (k in 1:K)
    beta[k] ~ normal(0, c[k] * lambda[k]^2); 
    
  // sample precision
  precision ~ gamma(a_tau, b_tau); 
  
  // Likelihood
  y ~ normal(X* beta, sqrt(1/precision));// likelihood
  
}


