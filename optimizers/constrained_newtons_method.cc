#include "optimizers/constrained_newtons_method.hh"

#include <iostream>

#include "optimizers/simple_newtons_method.hh"

namespace optimize {
namespace {
using CNM = ConstrainedNewtonsMethod;

Eigen::VectorXd
evaluate_constraints(const std::vector<const Constraint *> &constraints,
                     const Eigen::VectorXd &x, const double penalty,
                     const Eigen::VectorXd &lambda) {
  Eigen::VectorXd constraint_value(constraints.size());
  for (int i = 0; i < static_cast<int>(constraints.size()); i++) {
    constraint_value(i) = constraints[i]->evaluate(x);
    if (constraints[i]->is_inequality_constraint()) {
      constraint_value(i) = std::max(constraint_value(i), -lambda(i) / penalty);
    }
  }
  return constraint_value;
}

Eigen::VectorXd
update_multipliers(const std::vector<const Constraint *> &constraints,
                   const Eigen::VectorXd &x, const double penalty,
                   const Eigen::VectorXd &lambda) {
  return lambda +
         penalty * evaluate_constraints(constraints, x, penalty, lambda);
}

// Builds the lagrangian of the form:
// L(x, lambda, mu) = f(x) + \sum_i (lambda * g_i(x) + mu / 2 * g_i(x)**2)
struct AugmentedLagrangian : public Optimizable {
public:
  AugmentedLagrangian(const Optimizable &f,
                      const std::vector<const Constraint *> &constraints,
                      const Eigen::VectorXd lambda, const double penalty)
      : f_(f), constraints_(constraints), lambda_(lambda), penalty_(penalty) {}

  double evaluate(const Eigen::VectorXd &x) const override {
    const Eigen::VectorXd constraint_value =
        evaluate_constraints(constraints_, x, penalty_, lambda_);
    return f_.evaluate(x) + lambda_.dot(constraint_value) +
           penalty_ * constraint_value.squaredNorm() / 2.0;
  }

  Eigen::VectorXd gradient(const Eigen::VectorXd &x) const override {
    Eigen::VectorXd grad = f_.gradient(x);
    const Eigen::VectorXd constraint_value =
        evaluate_constraints(constraints_, x, penalty_, lambda_);
    for (int i = 0; i < static_cast<int>(constraints_.size()); i++) {
      const bool is_inactive_inequality_constraint =
          constraints_[i]->is_inequality_constraint() &&
          constraints_[i]->evaluate(x) < constraint_value(i);
      if (is_inactive_inequality_constraint) {
        continue;
      }
      const double c_val = constraint_value(i);
      const Eigen::VectorXd c_grad = constraints_[i]->gradient(x);
      grad += lambda_[i] * c_grad + penalty_ * c_val * c_grad;
    }
    return grad;
  }

  Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override {
    Eigen::MatrixXd hess = f_.hessian(x);
    const Eigen::VectorXd constraint_value =
        evaluate_constraints(constraints_, x, penalty_, lambda_);
    for (int i = 0; i < static_cast<int>(constraints_.size()); i++) {
      const bool is_inactive_inequality_constraint =
          constraints_[i]->is_inequality_constraint() &&
          constraints_[i]->evaluate(x) < constraint_value(i);
      if (is_inactive_inequality_constraint) {
        continue;
      }
      const double c_val = constraints_[i]->evaluate(x);
      const Eigen::VectorXd c_grad = constraints_[i]->gradient(x);
      const Eigen::MatrixXd c_hess = constraints_[i]->hessian(x);
      hess += lambda_[i] * c_hess +
              penalty_ * (c_val * c_hess + c_grad * c_grad.transpose());
    }
    return hess;
  }

  bool has_gradient() const override { return true; }
  bool has_hessian() const override { return true; }

private:
  const Optimizable &f_;
  const std::vector<const Constraint *> &constraints_;
  const Eigen::VectorXd lambda_;
  const double penalty_;
};

} // namespace

CNM::Result CNM::optimize(const Optimizable &f, const Eigen::VectorXd &x_init,
                          const std::vector<const Constraint *> &constraints,
                          const Options &opts) {
  constexpr double PENALTY_MULT = 10.0;
  const int num_constraints = constraints.size();
  double penalty = 0.1;
  Eigen::VectorXd lambda = 0.0 * Eigen::VectorXd::Ones(num_constraints);
  Eigen::VectorXd current_x = x_init;

  const SimpleNewtonsMethod::Options snm_opts = {
      .alpha = opts.alpha,
      .ftol = opts.ftol,
      .max_iters = opts.max_iters,
  };

  SimpleNewtonsMethod solver;
  for (int i = 0; i < 20; i++) {
    const AugmentedLagrangian new_cost(f, constraints, lambda, penalty);
    const auto result = solver.optimize(new_cost, current_x, snm_opts);
    current_x = result.optimum;
    std::cout << "iter: " << i << " current_x: " << current_x.transpose()
              << " curr penalty: " << penalty
              << " current lambda: " << lambda.transpose() << std::endl;
    lambda = update_multipliers(constraints, result.optimum, penalty, lambda);
    penalty *= PENALTY_MULT;
  }
  return {
      .cost = f.evaluate(current_x), .optimum = current_x, .converged = true};
}
} // namespace optimize
