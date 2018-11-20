#include "optimizers/simple_newtons_method.hh"

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

  Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const {
    return 4.0 * Eigen::MatrixXd::Identity(x.rows(), x.rows());
  }
};
} // namespace
TEST(SimpleNewtonsMethodTest, Quadratic1D) {
  const TestFunction f;
  SimpleNewtonsMethod optimizer;
  const Eigen::VectorXd test_point = Eigen::VectorXd::Ones(1) * 10.0;
  const Eigen::VectorXd optimum_point = Eigen::VectorXd::Ones(1) * -0.75;
  const SimpleNewtonsMethod::Options opts{
      .alpha = 1.0,
      .ftol = 1e-8,
      .max_iters = 50,
      .callback = [](const int iter, const SimpleNewtonsMethod::Result &step) {
        std::cout << iter << " cost: " << step.cost
                  << " x: " << step.optimum.transpose() << std::endl;
      }};
  const auto result = optimizer.optimize(f, test_point, opts);
  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.cost, f.evaluate(optimum_point) + opts.ftol);
}
} // namespace optimize
