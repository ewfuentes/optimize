#include <Eigen/Core>

namespace optimize {
template <int SIZE> struct Function {
  using Vector = Eigen::Matrix<double, SIZE, 1>;

  // Evaluate the function at the specified point
  virtual double operator()(const Vector &x);
};

template <int SIZE> struct DifferentiableFunction : public Function<SIZE> {
  using Vector = typename Function<SIZE>::Vector;
  virtual Vector gradient(const Vector &x);
};

template <int SIZE>
struct TwiceDifferentiableFunction : public DifferentiableFunction<SIZE> {
  using typename Function<SIZE>::Vector;
  using Hessian = Eigen::Matrix<double, SIZE, SIZE>;
  virtual Hessian hessian(const Vector &x);
};

template <> struct Function<1> { using Vector = double; };

template <> struct TwiceDifferentiableFunction<1> : DifferentiableFunction<1> {
  using Hessian = double;
};

} // namespace optimize
