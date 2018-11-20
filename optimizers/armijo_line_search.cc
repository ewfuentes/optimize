#include "optimizers/armijo_line_search.hh"

namespace optimize {

ArmijoResult armijo_line_search(const Optimizable &f,
                                const ArmijoPositionInfo &pos_info,
                                const ArmijoOpts &opts) {
  const double scaled_slope =
      -opts.sigma * pos_info.grad.transpose() * pos_info.step_direction;
  double current_step_size = opts.initial_step_size;
  for (int k = 0;; k++) {

    // evaluate the cost at the current step size
    const Eigen::VectorXd delta = current_step_size * pos_info.step_direction;
    const Eigen::VectorXd x_new = pos_info.x + delta;
    const double new_cost = f.evaluate(x_new);

    // Compute the change in cost if this step is accepted
    const double potential_cost_delta = pos_info.cost - new_cost;

    // Compute the cost threshold
    const double cost_threshold = scaled_slope * current_step_size;

    if (potential_cost_delta > cost_threshold) {
      return ArmijoResult{.k = k,
                          .step_size = current_step_size,
                          .cost = new_cost,
                          .x_new = x_new};
    } else {
      current_step_size *= opts.beta;
    }
  }
}
} // namespace optimize
