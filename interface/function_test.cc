#include "interface/function.hh"

#include <gtest/gtest.h>

namespace optimize {
TEST(FunctionTest, test_function) {
  struct TestFunction : public Function<1> {
    double operator()(const Vector &x) const override { return x + 1.0; }
  };
  TestFunction x;
  EXPECT_EQ(x(1), 2.0);
}

TEST(DifferentiableFunctionTest, test_function) {
  struct TestFunction : public DifferentiableFunction<1> {
    double operator()(const double &x) const override { return x + 1.0; }

    double gradient(const double &x) const override { return 1.0; }
  };
  TestFunction x;
  EXPECT_EQ(x(1), 2.0);
  EXPECT_EQ(x.gradient(1.0), 1.0);
}

TEST(TwiceDifferentiableFunctionTest, test_function) {
  struct TestFunction : public TwiceDifferentiableFunction<1> {
    double operator()(const double &x) const override { return x * x + 2 * x + 1.0; }

    double gradient(const double &x) const override { return 2 * x + 2; }

    double hessian(const double &x) const override { return 2; }
  };
  TestFunction x;
  EXPECT_EQ(x(1), 4.0);
  EXPECT_EQ(x.gradient(1.0), 4.0);
  EXPECT_EQ(x.hessian(1.0), 2.0);
}
} // namespace optimize
