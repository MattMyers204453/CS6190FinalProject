import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from numpy.linalg import inv
from scipy.stats import multivariate_normal

# Combine datasets from different seasons
X_TRAIN_1 = np.load("X-TRAIN-1.npy")
Y_TRAIN_1 = np.load("Y-TRAIN-1.npy")

X_TRAIN_2 = np.load("X-TRAIN-2.npy")
Y_TRAIN_2 = np.load("Y-TRAIN-2.npy")

X_TRAIN = np.vstack((X_TRAIN_1, X_TRAIN_2))
Y_TRAIN = np.concatenate((Y_TRAIN_1, Y_TRAIN_2))

X_TEST= np.load("X-TEST.npy")
Y_TEST = np.load("Y-TEST.npy")

train_size = X_TRAIN.shape[0]
print(f"TRAINING SIZE: {train_size}")

test_size = X_TEST.shape[0]
print(f"TESTING SIZE: {test_size}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_likelihood(coefficients, X, Y):
    linear_term = X @ coefficients
    log_likelihood = np.sum(Y * np.log(sigmoid(linear_term)) + (1 - Y) * np.log(1 - sigmoid(linear_term)))
    return log_likelihood

def neg_log_posterior(coefficients, X, Y, prior_mean, prior_covariance):
    prior_precision = inv(prior_covariance)
    prior = -0.5 * (coefficients - prior_mean).T @ prior_precision @ (coefficients - prior_mean) # multivariate Gaussian
    likelihood = log_likelihood(coefficients, X, Y)
    return -1 * (prior + likelihood)

# From https://users.cs.utah.edu/~zhe/teach/pdf/8-generalized-linear.pdf
def hessian_inv(w, x):
    y = sigmoid(x @ w)
    R = np.diag(y * (1 - y)) 
    hessian = x.T @ R @ x + np.eye(x.shape[1])  #Second gradient of Prior may be wrong here
    return inv(hessian)


def bayesian_logistic_regression(X_train, Y_train):

    # Initialize prior to multivariate standard Gaussian
    prior_mean = np.zeros(X_train.shape[1])
    prior_covariance = np.eye(X_train.shape[1])
    # prior_mean = np.ones(X_train.shape[1]) * 0.5
    # prior_covariance = np.eye(X_train.shape[1]) * 2

    # Loop through training examples and update prior to be the posterior of the previous step
    for i in range(len(X_train)):
        x = X_train[:i+1, :]
        y = Y_train[:i+1]

        # Laplace Approximation of posterior
        result = minimize(neg_log_posterior, prior_mean, args=(x, y, prior_mean, prior_covariance), method="BFGS")
        posterior_mean = result.x
        posterior_covariance = result.hess_inv
        # posterior_covariance = hessian_inv(posterior_mean, x)

        # Use this Laplace-approximated posterior as the prior in the next step
        prior_mean = posterior_mean
        prior_covariance = posterior_covariance

    return posterior_mean, posterior_covariance

# Logistic regression prediction function that does not incorporate uncertainty (point estimate, MAP)
def predict(x, coefficients):
    linear_term = x @ coefficients
    prob = sigmoid(linear_term)
    return np.where(prob >= 0.5, 1, 0)

# Logistic regression prediction function that does incorporate uncertainty (weighted average over posterior)
def predict_with_posterior_predictive_dist(X_test, posterior_mean, posterior_covariance, num_samples=500):
    # Generate samples from posterior distribution
    samples = multivariate_normal.rvs(mean=posterior_mean, cov=posterior_covariance, size=num_samples)

    # Calculate average predictive probability over all samples
    prob_sum = np.zeros(X_test.shape[0])
    for s in samples:
        prob_sum += sigmoid(X_test @ s)

    prob_avg = prob_sum / num_samples

    # Predict using average probability
    return np.where(prob_avg >= 0.5, 1, 0)

# Guess outcome of each game
def flip_coin_baseline_model(X_test):
    np.random.seed(5)
    return np.random.choice([0, 1], size=len(X_test))

# Train model
posterior_mean, posterior_covariance = bayesian_logistic_regression(X_TRAIN, Y_TRAIN)
print(posterior_mean)
print(posterior_covariance)

# Test model
predictions = predict(X_TEST, posterior_mean)
accuracy = sum(predictions == Y_TEST) / len(Y_TEST)
print(f"Bayesian logistic regression model accuracy: {(accuracy * 100):.2f}%")

# Test model using predictive distribution (using standard Monte Carlo sampling)
predictions = predict_with_posterior_predictive_dist(X_TEST, posterior_mean, posterior_covariance)
accuracy = sum(predictions == Y_TEST) / len(Y_TEST)
print(f"Bayesian Logistic Regression with posterior predictive distribution accuracy: {(accuracy * 100):.2f}%")

# Get baseline performance
random_predictions = flip_coin_baseline_model(X_TEST)
random_accuracy = np.mean(random_predictions == Y_TEST)
print(f"Baseline guessing model accuracy: {(random_accuracy * 100):.2f}%")
