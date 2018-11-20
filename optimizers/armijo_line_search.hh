#pragma once

#include <Eigen/Core>

#include "optimizers/optimizer.hh"

namespace optimize {

struct ArmijoResult {
  // smallest admissible `k`
  int k;
  // selected step size
  double step_size;
  // cost at the new position
  double cost;
  // Selected point
  Eigen::VectorXd x_new;
};

struct ArmijoOpts {
  // The slope rescaling factor
  double sigma = 1e-2;
  // The exponential backoff factor
  double beta = 0.5;
  // Initial step size
  double initial_step_size = 1.0;
};

struct ArmijoPositionInfo {
  // Position where the still will be made from
  Eigen::VectorXd x;
  // Direction and magnitude of steepest ascent
  Eigen::VectorXd grad;
  // Direction in which the line search is being performed.
  // Will be normalized inside of the function
  Eigen::VectorXd step_direction;
  // Cost at X
  double cost;
};

// Performs a line search according to the Armijo Rule
// Given:
// - A cost function `f`
// - A position about which the line search is starting `x_0`
// - The gradient of the cost function at this point `grad_f`
// - The direction in which the line search is to be performed `d`
// and the parameters:
// - gradient multiplier `sigma` (must be less than 1, ideally [1e-1, 1e-5])
// - step multiplier `beta` (must be less than 1, ideally [0.1, 0.5])
// - initial step size `s`
// Select the smallest `k` such that
// f(x) - f(x + beta**k * s * d) >= -sigma * beta**k * s * grad_f' * d
//
// In order words, the change in cost between the current x and the new position
// is greater than gradient in cost along the step direction times the
// computed step size times a scaling factor
ArmijoResult armijo_line_search(const Optimizable &f,
                                const ArmijoPositionInfo &pos_info,
                                const ArmijoOpts &opts);

} // namespace optimize
