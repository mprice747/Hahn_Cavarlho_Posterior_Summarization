
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // predictor matrix
  vector[N] y; // outcome vector
  real<lower=0> a_beta; // prior precision beta, (alpha parameter)
  real<lower=0> b_beta; // prior precision beta, (beta parameter)
  real<lower=0> a_tau; // prior precision, (alpha parameter)
  real<lower=0> b_tau; // prior precison, (beta parameter)
  real<lower=0, upper=1> v_0; // Small number near 0
}

 // The parameters accepted by the model, beta/precision/w
parameters {
  vector[K] beta; // Coefficients from model 
  real<lower=0> precision; // Precision Parameter
  real<lower=0, upper=1> w; // hyperprior for Bernoullis
}

model {
  
  // Sample w
  w ~ beta(1, 1); 
  
  // Sample precision
  precision ~ gamma(a_tau, b_tau);
  
  // Sample from mixed t
  for (k in 1:K)
    target += log_mix(w,
                    student_t_lpdf(beta[k] | 2 * a_beta, 0, sqrt(v_0 * b_beta/a_beta)),
                    student_t_lpdf(beta[k] | 2 * a_beta, 0, sqrt(b_beta/a_beta)));
                    
                    
  y ~ normal(X* beta, sqrt(1/precision));
                    
                    
  
}

