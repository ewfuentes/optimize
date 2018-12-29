#include "optimizers/constrained_newtons_method.hh"

#include <iostream>

#include "optimizers/simple_newtons_method.hh"

namespace optimize {
namespace {
using CNM = ConstrainedNewtonsMethod;

// Builds the lagrangian of the form:
// L(x, lambda, mu) = f(x) + \sum_i (lambda * g_i(x) + mu / 2 * g_i(x)**2)
struct AugmentedLagrangian : public Optimizable {
public:
  AugmentedLagrangian(const Optimizable &f,
                      const std::vector<const Constraint *> &constraints,
                      const Eigen::VectorXd lambda, const Eigen::VectorXd mu)
      : f_(f), constraints_(constraints), lambda_(lambda), mu_(mu) {}
  Eigen::VectorXd evaluate_constraints(const Eigen::VectorXd &x) const {
    Eigen::VectorXd constraint_value(constraints_.size());
    for (int i = 0; i < static_cast<int>(constraints_.size()); i++) {
      constraint_value(i) = constraints_[i]->evaluate(x);
    }
    return constraint_value;
  }

  double evaluate(const Eigen::VectorXd &x) const override {
    const Eigen::VectorXd constraint_value = evaluate_constraints(x);
    return f_.evaluate(x) + lambda_.dot(constraint_value) +
           mu_.dot(constraint_value.cwiseProduct(constraint_value));
  }

  Eigen::VectorXd gradient(const Eigen::VectorXd &x) const override {
    Eigen::VectorXd grad = f_.gradient(x);
    for (int i = 0; i < static_cast<int>(constraints_.size()); i++) {
      const double constraint_value = constraints_[i]->evaluate(x);
      const Eigen::VectorXd constraint_gradient = constraints_[i]->gradient(x);
      grad += lambda_[i] * constraint_gradient +
              mu_[i] * constraint_value * constraint_gradient;
    }
    return grad;
  }

  Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override {
    Eigen::MatrixXd hess = f_.hessian(x);
    for (int i = 0; i < static_cast<int>(constraints_.size()); i++) {
      const double c_val = constraints_[i]->evaluate(x);
      const Eigen::VectorXd c_grad = constraints_[i]->gradient(x);
      const Eigen::MatrixXd c_hess = constraints_[i]->hessian(x);
      hess += lambda_[i] * c_hess +
              mu_[i] * (c_val * c_hess + c_grad * c_grad.transpose());
    }
    return hess;
  }

  bool has_gradient() const override { return true; }
  bool has_hessian() const override { return true; }

private:
  const Optimizable &f_;
  const std::vector<const Constraint *> &constraints_;
  const Eigen::VectorXd lambda_;
  const Eigen::VectorXd mu_;
};
} // namespace

CNM::Result CNM::optimize(const Optimizable &f, const Eigen::VectorXd &x_init,
                          const std::vector<const Constraint *> &constraints,
                          const Options &opts) {
  const int num_constraints = constraints.size();
  Eigen::VectorXd mu = 0.1 * Eigen::VectorXd::Ones(num_constraints);
  Eigen::VectorXd lambda = 0.0 * Eigen::VectorXd::Ones(num_constraints);
  Eigen::VectorXd current_x = x_init;
  constexpr double MU_MULT = 4.0;

  const SimpleNewtonsMethod::Options snm_opts = {
      .alpha = opts.alpha,
      .ftol = opts.ftol,
      .max_iters = opts.max_iters,
  };

  SimpleNewtonsMethod solver;
  for (int i = 0; i < 20; i++) {
    const AugmentedLagrangian new_cost(f, constraints, lambda, mu);
    const auto result = solver.optimize(new_cost, current_x, snm_opts);
    current_x = result.optimum;
    mu *= MU_MULT;
    lambda = lambda + mu.cwiseProduct(new_cost.evaluate_constraints(result.optimum));
    std::cout << "iter: " << i
              << " current_x: " << current_x.transpose()
              << " curr mu: " << mu
              << " current lambda: " << lambda
              << " Violation: " <<  mu.cwiseProduct(new_cost.evaluate_constraints(result.optimum))
              << std::endl;
  }
  return {.cost = f.evaluate(current_x), .optimum = current_x};
}
} // namespace optimize
