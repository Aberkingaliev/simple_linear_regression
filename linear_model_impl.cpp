#include "linear_model.h"
#include <cmath>
#include <vector>

LinearModel::LinearModel(int l) {
  weights = std::vector<double>(l, 0.0);
  bias = 0.0;
}

LinearRegression::LinearRegression(int l) : LinearModel(l) {
  weights = std::vector<double>(l, 0.0);
  bias = 0.0;
};

void LinearRegression::train(std::vector<std::vector<double>> &X,
                             std::vector<double> &y, double learning_rate,
                             double epochs) {
  size_t sample_size = X.size();

  for (size_t ep = 0; ep < epochs; ++ep) {
    for (size_t i = 0; i < sample_size; ++i) {
      double predicted = predict(X[i]);
      double error = (predicted - y[i]);

      for (size_t j = 0; j < weights.size(); ++j) {
        weights[j] -= learning_rate * error * X[i][j];
      }

      bias -= learning_rate * error;
    }
  }
};

double LinearRegression::predict(std::vector<double> &X) {
  double result = bias;
  size_t feature_size = X.size();

  for (size_t i = 0; i < feature_size; ++i) {
    result += X[i] * weights[i];
  }

  return result;
};