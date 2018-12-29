#pragma once

#include <memory>

#include "optimizers/optimizer.hh"

namespace optimize {
struct Constraint : public Optimizable {
  virtual bool is_equality_constraint() const { return true; };
};

struct ConstraintSet {
  std::vector<std::unique_ptr<Constraint>> constraints;
};
} // namespace optimize
