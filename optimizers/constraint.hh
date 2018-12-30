#pragma once

#include <memory>

#include "optimizers/optimizer.hh"

namespace optimize {
struct Constraint : public Optimizable {
  virtual bool is_inequality_constraint() const { return false; };
};

struct ConstraintSet {
  std::vector<std::unique_ptr<Constraint>> constraints;
};
} // namespace optimize
