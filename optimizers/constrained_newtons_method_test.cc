#include "optimizers/constrained_newtons_method.hh"

#include <gtest/gtest.h>

#include "optimizers/optimizer.hh"

namespace optimize {
namespace {
struct TestFunction : public Optimizable {
  double evaluate(const Eigen::VectorXd &x) const {
    return (2 * x.transpose() * x + 3 * Eigen::RowVectorXd::Ones(x.rows()) * x +
            5.0 * Eigen::VectorXd::Ones(1))(0);
  }

  Eigen::VectorXd gradient(const Eigen::VectorXd &x) const {
    return 4 * x + 3.0 * Eigen::VectorXd::Ones(x.rows());
  }

  Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const {
    return 4.0 * Eigen::MatrixXd::Identity(x.rows(), x.rows());
  }
};

// implements (x_1 + x_2 = 0)
struct TestEqConstraint : public Constraint {
  double evaluate(const Eigen::VectorXd &x) const override {
    return x(0) - x(1);
  }

  Eigen::VectorXd gradient(const Eigen::VectorXd &x) const override {
    Eigen::Vector2d out = Eigen::Vector2d::Ones();
    out(1) = -1;
    return out;
  }

  Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override {
    return Eigen::MatrixXd::Zero(2, 2);
  }

  bool has_gradient() const override { return true; }
  bool has_hessian() const override { return true; }
  bool is_inequality_constraint() const override { return false; }
};

// implements (x_1 + x_2 < 0)
struct TestIneqConstraint : public Constraint {
  double evaluate(const Eigen::VectorXd &x) const override {
    return x(0) + x(1);
  }

  Eigen::VectorXd gradient(const Eigen::VectorXd &x) const override {
    Eigen::Vector2d out = Eigen::Vector2d::Ones();
    out(1) = -1;
    return out;
  }

  Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override {
    return Eigen::MatrixXd::Zero(2, 2);
  }

  bool has_gradient() const override { return true; }
  bool has_hessian() const override { return true; }
  bool is_inequality_constraint() const override { return true; }
};
} // namespace

TEST(ConstrainedNewtonsMethodTest, Quadratic2DEqualityOnConstrainedMinimum) {
  const TestFunction f;
  ConstrainedNewtonsMethod optimizer;
  const Eigen::VectorXd test_point = Eigen::VectorXd::Ones(2) * 10.0;
  const Eigen::VectorXd optimum_point = Eigen::VectorXd::Ones(2) * -0.75;
  const ConstrainedNewtonsMethod::Options opts{
      .alpha = 1.0,
      .ftol = 1e-8,
      .max_iters = 50,
      .callback = [](const int iter,
                     const ConstrainedNewtonsMethod::Result &step) {
        std::cout << iter << " cost: " << step.cost
                  << " x: " << step.optimum.transpose() << std::endl;
      }};
  const TestEqConstraint c;
  std::vector<const Constraint *> constraints;
  constraints.push_back(&c);
  const auto result = optimizer.optimize(f, test_point, constraints, opts);
  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.cost, f.evaluate(optimum_point) + opts.ftol);
}
} // namespace optimize
