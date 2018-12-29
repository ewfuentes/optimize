#pragma once

#include <Eigen/Core>

namespace optimize {

struct Optimizable {
  virtual double evaluate(const Eigen::VectorXd &x) const = 0;
  virtual Eigen::VectorXd gradient(const Eigen::VectorXd &x) const { return x; };
  virtual Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const { return x; };

  virtual bool has_gradient() const { return false; }
  virtual bool has_hessian() const { return false; }
};

} // namespace optimize
