data {
  int<lower=1> N;             // Number of observations
  vector[N] x;                // Predictor variable
  int<lower=0, upper=1> g[N]; // Group indicator (binary)
  vector[N] y;                // Observed outcome
}

parameters {
  real beta0;                 // Intercept for mean
  real beta1;                 // Coefficient for x
  real<lower=0> delta0;       // Intercept for standard deviation
  real<lower=0> delta1;       // Effect of group indicator on standard deviation
}

model {
  vector[N] mu;               // Mean of y
  vector[N] sigma;            // Standard deviation of y

  // Define mean and standard deviation
  mu = beta0 + beta1 * x;
  sigma = delta0 + delta1 * to_vector(g);

  // Exotic priors
  beta0 ~ cauchy(0, 5);   // wide heavy-tailed Cauchy prior centered at 0
  beta1 ~ cauchy(0, 1);   // narrower  Heavy-tailed Cauchy prior centered at 0
  delta0 ~ gamma(2, 2);   // Gamma prior (shape=2, rate=2), ensuring positivity
  delta1 ~ gamma(2, 2);   // Gamma prior (shape=2, rate=2), ensuring positivity

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_pred;           // Predicted outcomes

  for (n in 1:N) {
    y_pred[n] = normal_rng(beta0 + beta1 * x[n], delta0 + delta1 * g[n]);
  }
}
