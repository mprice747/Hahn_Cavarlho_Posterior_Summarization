
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  vector[N] y;      // outcome vector
  real<lower=0> a_tau; // prior precision alpha
  real<lower=0> b_tau; // prior precision beta
  vector<lower=0>[K] sigma_prior; // beta prior SD (small)
  vector<lower=0>[K] c_prior; // Large Number 
  vector<lower=0>[K] pi_prior; // Prior for gamma, usually 1/2
}

// The parameters accepted by the model, beta/precision
parameters {
  vector[K] beta;       // coefficients for predictors
  real<lower=0> precision; // variance 
}

// SSVS Model
model {
  

precision ~ gamma(a_tau, b_tau);
// Mixed Normal Prior 
 for (k in 1:K)
  target += log_mix(pi_prior[k],
                    normal_lpdf(beta[k] | 0, c_prior[k] * sigma_prior[k]),
                    normal_lpdf(beta[k] | 0, sigma_prior[k]));
    
  y ~ normal(X* beta, sqrt(1/precision));  
}


