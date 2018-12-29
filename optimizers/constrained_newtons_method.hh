#pragma once

#include <memory>

#include "optimizers/constraint.hh"
#include "optimizers/optimizer.hh"

namespace optimize {
class ConstrainedNewtonsMethod {
public:
  struct Result;
  using IterationCallback = std::function<void(const int, const Result &)>;

  struct Options {
    double alpha = 1e-1;
    double ftol = 1e-5;
    int max_iters = 1000;
    IterationCallback callback = [](const int, const Result &) {};
  };

  struct Result {
    double cost = -1;
    Eigen::VectorXd optimum{};
    bool converged = false;
  };

  Result optimize(const Optimizable &f, const Eigen::VectorXd &x_init,
                  const std::vector<const Constraint *> &constraints,
                  const Options &opt);
};
} // namespace optimize
