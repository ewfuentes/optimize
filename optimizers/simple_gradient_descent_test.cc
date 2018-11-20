#include "optimizers/simple_gradient_descent.hh"

#include <gtest/gtest.h>

#include "optimizers/optimizer.hh"

namespace optimize {
namespace {
struct TestFunction : public Optimizable {
  double evaluate(const Eigen::VectorXd &x) const {
    return (2 * x.transpose() * x + 3 * x +
            5.0 * Eigen::VectorXd::Ones(x.rows()))(0);
  }

  Eigen::VectorXd gradient(const Eigen::VectorXd &x) const {
    return 4 * x + 3.0 * Eigen::VectorXd::Ones(x.rows());
  }
};
} // namespace
TEST(SimpleGradientDescentTest, Quadratic1D) {
  const TestFunction f;
  SimpleGradientDescent optimizer;
  const Eigen::VectorXd test_point = Eigen::VectorXd::Ones(1) * 10.0;
  const Eigen::VectorXd optimum_point = Eigen::VectorXd::Ones(1) * -0.75;
  const SimpleGradientDescent::Options opts{
      .alpha = 1e-1,
      .ftol = 1e-8,
      .max_iters = 50,
  };
  const auto result = optimizer.optimize(f, test_point, opts);
  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.cost, f.evaluate(optimum_point) + opts.ftol);
}
} // namespace optimize
