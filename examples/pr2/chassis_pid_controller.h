// Copyright 2019 Toyota Research Institute. All rights reserved.
#pragma once

#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/examples/pr2/robot_parameters.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace pr2 {

class ChassisPidController final : public systems::LeafSystem<double> {
 public:
  static constexpr const char* kPartName = "chassis";

  /// A convenient struct that contains all the necessary information to compute
  /// a PID control effort of a particular joint.
  struct JointControlInfo {
    std::string joint_name;
    double kp{0.0};
    double kd{0.0};
    double ki{0.0};
    double effort_limit{0.0};
    int position_index{0};
    // Velocity index w.r.t the full state vector. To get the velocity index
    // w.r.t the velocity only vector. Use this index minus the total number of
    // positions of the plant.
    int velocity_index{0};
  };

  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ChassisPidController)

  ChassisPidController(const multibody::MultibodyPlant<double>& robot_plant,
                       const RobotParameters& parameters);

  /// @name Named accessors for this System's input and output ports.
  //@{
  const systems::InputPort<double>& get_desired_state_input_port() const {
    DRAKE_DEMAND(input_port_desired_state_ != nullptr);
    return *input_port_desired_state_;
  }

  const systems::InputPort<double>& get_estimated_state_input_port() const {
    DRAKE_DEMAND(input_port_estimated_state_ != nullptr);
    return *input_port_estimated_state_;
  }

  const systems::OutputPort<double>& get_generalized_force_output_port() const {
    DRAKE_DEMAND(output_port_generalized_force_ != nullptr);
    return *output_port_generalized_force_;
  }
  //@}

 private:
  systems::BasicVector<double> AllocateOutput() const {
    return systems::BasicVector<double>(num_velocities_);
  }

  virtual void CalcOutput(const systems::Context<double>& context,
                          systems::BasicVector<double>* output) const;

  const int num_positions_;
  const int num_velocities_;

  using PartControlInfo = std::vector<JointControlInfo>;
  std::vector<PartControlInfo> parts_control_info_;

  const systems::InputPort<double>* input_port_desired_state_{};
  const systems::InputPort<double>* input_port_estimated_state_{};
  const systems::OutputPort<double>* output_port_generalized_force_{};
};

}  // namespace pr2
}  // namespace examples
}  // namespace drake
