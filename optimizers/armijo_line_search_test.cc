#include "optimizers/armijo_line_search.hh"

#include <gtest/gtest.h>

#include "optimizers/optimizer.hh"

namespace optimize {
namespace {
struct TestFunction : public Optimizable {
  double evaluate(const Eigen::VectorXd &x) const {
    return (1.0 * x.transpose() * x - 10.0 * Eigen::RowVectorXd::Ones(x.rows()) * x +
            25 * Eigen::VectorXd::Ones(1))(0);
  }

  Eigen::VectorXd gradient(const Eigen::VectorXd &x) const {
    return 2.0 * x - 10.0 * Eigen::VectorXd::Ones(x.rows());
  }

  Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const {
    return 2.0 * Eigen::MatrixXd::Identity(x.rows(), x.rows());
  }
};
} // namespace

TEST(ArmijoLineSearchTest, happy_case) {
  // Setup
  // Test point
  const TestFunction f;
  const Eigen::VectorXd test_point = Eigen::VectorXd::Ones(1) * -10.0;
  const Eigen::VectorXd grad_at_point = f.gradient(test_point);
  const Eigen::VectorXd step_direction = Eigen::VectorXd::Ones(1);

  const ArmijoPositionInfo pos{
      .x = test_point,
      .grad = grad_at_point,
      .step_direction = step_direction,
      .cost = f.evaluate(test_point),
  };

  const ArmijoOpts opts{
      .sigma = 0.1,
      .beta = 0.75,
      .initial_step_size = 60.0,
  };

  // Action
  const ArmijoResult result = armijo_line_search(f, pos, opts);

  // Verify
  // For this cost function and selected options, the admissible step sizes are:
  // Cost at test point = 225
  // Grad at test point = -30
  // scaled gradient = -3
  // -3 (x + 10) + 225 = x**2 - 10 x + 25
  // -3 x + 195 = x**2 - 10 x + 25
  // 0 = x**2 - 7 x -170
  // x = -10, 17
  // Acceptable steps -> [0, 27]
  // Given that the initial step size is 60 and the back off is 0.75, we'll need
  // iterate three times so that the step size is 25.3125
  constexpr double EXPECTED_STEP_SIZE = 25.3125;
  const Eigen::VectorXd expected_x = Eigen::VectorXd::Ones(1) * 15.3125;
  constexpr double TOL = 1e-5;
  EXPECT_EQ(result.k, 3);
  EXPECT_NEAR(result.step_size, EXPECTED_STEP_SIZE, TOL);
  EXPECT_NEAR(result.cost, f.evaluate(expected_x), TOL);
  EXPECT_NEAR(result.x_new(0), expected_x(0), TOL);
}
} // namespace optimize
