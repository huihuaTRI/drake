#include "drake/geometry/optimization/convex_set.h"

#include <algorithm>
#include <limits>
#include <memory>

namespace drake {
namespace geometry {
namespace optimization {

ConvexSet::ConvexSet(
    std::function<std::unique_ptr<ConvexSet>(const ConvexSet&)> cloner,
    int ambient_dimension)
    : cloner_(std::move(cloner)), ambient_dimension_(ambient_dimension) {
  DRAKE_DEMAND(ambient_dimension >= 0);
}

ConvexSet::~ConvexSet() = default;

std::unique_ptr<ConvexSet> ConvexSet::Clone() const { return cloner_(*this); }

std::vector<solvers::Binding<solvers::Constraint>>
ConvexSet::AddPointInNonnegativeScalingConstraints(
    solvers::MathematicalProgram* prog,
    const Eigen::Ref<const solvers::VectorXDecisionVariable>& x,
    const symbolic::Variable& t) const {
  DRAKE_DEMAND(x.size() == ambient_dimension());
  std::vector<solvers::Binding<solvers::Constraint>> constraints =
      DoAddPointInNonnegativeScalingConstraints(prog, x, t);
  constraints.emplace_back(prog->AddBoundingBoxConstraint(
      0, std::numeric_limits<double>::infinity(), t));
  return constraints;
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
