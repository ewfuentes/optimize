#pragma once

#include <Eigen/Core>

namespace optimize {
template <int SIZE> struct Function {
  using Vector = Eigen::Matrix<double, SIZE, 1>;
  static constexpr double DIMENSION = SIZE;

  // Evaluate the function at the specified point
  virtual double operator()(const Vector &x) const = 0;
};

template <int SIZE> struct DifferentiableFunction : public Function<SIZE> {
  using Vector = typename Function<SIZE>::Vector;
  virtual Vector gradient(const Vector &x) const = 0;
};

template <int SIZE>
struct TwiceDifferentiableFunction : public DifferentiableFunction<SIZE> {
  using typename Function<SIZE>::Vector;
  using Hessian = Eigen::Matrix<double, SIZE, SIZE>;
  virtual Hessian hessian(const Vector &x) const = 0;
};

template <> struct Function<1> {
  using Vector = double;
  virtual double operator()(const double &x) const = 0;
};

  template <> struct DifferentiableFunction<1> : public Function<1> {
  using Vector = double;
  virtual double gradient(const double &x) const = 0;
};

template <> struct TwiceDifferentiableFunction<1> : public DifferentiableFunction<1> {
  using Hessian = double;
  virtual double hessian(const double &x) const = 0;
};

} // namespace optimize
