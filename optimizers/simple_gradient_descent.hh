#pragma once

#include <functional>
#include <limits>

#include "optimizers/optimizer.hh"

namespace optimize {
class SimpleGradientDescent {
public:
  struct Result;
  using IterationCallback = std::function<void(const int, const Result &)>;

  struct Options {
    // How much the gradient gets scaled by when taking a step
    double alpha = 1e-2;

    // Terminate when the difference between the current and the previous
    // function evaluation differ by less than this much
    double ftol = 1e-5;

    // Maximum number of iterations
    int max_iters = 1000;

    IterationCallback callback = [](const int, const Result &) {};
    ;
  };

  struct Result {
    double cost = -1;
    Eigen::VectorXd optimum{};
    bool converged = false;
  };

  Result optimize(const Optimizable &f, const Eigen::VectorXd &x_init,
                  const Options &opt);
};
} // namespace optimize
