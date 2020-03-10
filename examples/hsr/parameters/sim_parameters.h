#pragma once

#include <string>

namespace drake {
namespace examples {
namespace hsr {
namespace parameters {

/// A structure to hold values that are constant at runtime after
/// command-line processing.  Access the sole instance via `flags()`.
struct SimParameters {
  double kp{};
  double kd{};
  double ki{};
  double gravity{};
  double target_realtime_rate{};
  double integration_accuracy{};
  double penetration_allowance{};
  double v_stiction_tolerance{};
  double inclined_plane_coef_static_friction{};
  double inclined_plane_coef_kinetic_friction{};
  double simulation_time{};
  double time_step{};

  bool use_constant_desired_state{};
};

/// @return a constref to the singleton `SimParameters` instance.
const SimParameters& hsr_sim_flags();

}  // namespace parameters
}  // namespace hsr
}  // namespace examples
}  // namespace drake
