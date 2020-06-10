#include "drake/examples/pr2/pr2_pd_controller.h"

#include "drake/common/drake_assert.h"
#include "drake/math/saturate.h"

namespace drake {
namespace examples {
namespace pr2 {

namespace systems = drake::systems;

std::vector<JointControlInfo> ParsePartJointControlInfoFromParameters(
    const multibody::MultibodyPlant<double>& robot_plant,
    const PartParameters& part_parameters) {
  // Parses and loads all the joint parameters that belong to the given parts.
  std::vector<JointControlInfo> part_joints_pid_info;
  for (const auto& joint_parameter : part_parameters.joints_parameters) {
    JointControlInfo joint_pid_info;
    joint_pid_info.joint_name = joint_parameter.name;
    joint_pid_info.kp = joint_parameter.gains.kp;
    joint_pid_info.kd = joint_parameter.gains.kd;
    joint_pid_info.ki = joint_parameter.gains.ki;
    joint_pid_info.effort_limit = joint_parameter.effort_limit;
    joint_pid_info.position_index =
        robot_plant.GetJointByName(joint_pid_info.joint_name).position_start();
    joint_pid_info.velocity_index =
        robot_plant.num_positions() +
        robot_plant.GetJointByName(joint_pid_info.joint_name).velocity_start();
    part_joints_pid_info.push_back(joint_pid_info);
  }

  return part_joints_pid_info;
}

Pr2PdController::Pr2PdController(
    const multibody::MultibodyPlant<double>& robot_plant,
    const PartParameters& part_parameters)
    : num_positions_(robot_plant.num_positions()),
      num_velocities_(robot_plant.num_velocities()) {
  DRAKE_DEMAND(num_positions_ > 0);
  DRAKE_DEMAND(num_velocities_ > 0);
  // Command state.
  const int state_size = num_positions_ + num_velocities_;
  input_port_desired_state_ = &this->DeclareVectorInputPort(
      "desired_state", systems::BasicVector<double>(state_size));
  // Estimate state.
  input_port_estimated_state_ = &this->DeclareVectorInputPort(
      "estimated_state", systems::BasicVector<double>(state_size));

  output_port_generalized_force_ = &this->DeclareVectorOutputPort(
      "generalized_force", systems::BasicVector<double>(num_velocities_),
      &Pr2PdController::CalcOutput);

  part_control_info_ =
      ParsePartJointControlInfoFromParameters(robot_plant, part_parameters);
}

void Pr2PdController::CalcOutput(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* output) const {
  output->SetZero();

  const bool hasEstimatedState =
      this->get_estimated_state_input_port().HasValue(context);
  const bool hasDesiredState =
      this->get_desired_state_input_port().HasValue(context);

  if (hasEstimatedState && hasDesiredState) {
    const auto& estimated_state =
        this->get_estimated_state_input_port().Eval(context);
    const auto& desired_state =
        this->get_desired_state_input_port().Eval(context);

    // Apply PD controller to the interested joints.
    for (const auto& joint_control_info : part_control_info_) {
      const int position_index = joint_control_info.position_index;
      const int velocity_index = joint_control_info.velocity_index;
      const double raw_torque =
          joint_control_info.kp * (desired_state[position_index] -
                                   estimated_state[position_index]) +
          joint_control_info.kd *
              (desired_state[velocity_index] - estimated_state[velocity_index]);
      (*output)[velocity_index - num_positions_] =
          drake::math::saturate(raw_torque, -joint_control_info.effort_limit,
                                joint_control_info.effort_limit);
    }
  }
}

}  // namespace pr2
}  // namespace examples
}  // namespace drake
