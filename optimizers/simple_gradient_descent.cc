#include "optimizers/simple_gradient_descent.hh"

namespace optimize {
namespace {
using SGD = SimpleGradientDescent;
} // namespace

SGD::Result SGD::optimize(const Optimizable &f, const Eigen::VectorXd &x_init,
                          const Options &opts) {
  Result out{f.evaluate(x_init), x_init, false};
  opts.callback(-1, out);

  for (int i = 0; i < opts.max_iters; i++) {
    // compute the gradient
    const Eigen::VectorXd grad = f.gradient(out.optimum);

    // compute the step
    const Eigen::VectorXd new_x = out.optimum - opts.alpha * grad;

    // compute the function value at the new step
    const double new_cost = f.evaluate(new_x);

    // check for convergence
    const bool converged = std::abs(out.cost - new_cost) < opts.ftol;
    out.cost = new_cost;
    out.optimum = new_x;
    out.converged = converged;

    opts.callback(i, out);

    if (converged) {
      break;
    }
  }
  return out;
}

} // namespace optimize
