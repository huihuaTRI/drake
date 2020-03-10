#include "drake/examples/hsr/sim_status_sender.h"

namespace drake {
namespace examples {
namespace hsr {

SimStatusSender::SimStatusSender(
    const drake::multibody::MultibodyPlant<double>* robot_plant)
    : robot_plant_(robot_plant) {
  DRAKE_DEMAND(robot_plant != nullptr);
  const int state_size =
      robot_plant_->num_positions() + robot_plant_->num_velocities();
  // Commanded state.
  this->DeclareInputPort("estimated_state", drake::systems::kVectorValued,
                         state_size);

  this->DeclareAbstractOutputPort("sim_status_message",
                                  &SimStatusSender::MakeOutput,
                                  &SimStatusSender::CalcOutput);
}

lcmt_hsr_sim_status SimStatusSender::MakeOutput() const {
  lcmt_hsr_sim_status command_message{};

  const int num_actuators = robot_plant_->num_actuators();
  command_message.num_joints = num_actuators;
  command_message.joint_name.resize(num_actuators);
  command_message.joint_position.resize(num_actuators);
  command_message.joint_velocity.resize(num_actuators);

  return command_message;
}

void SimStatusSender::CalcOutput(const systems::Context<double>& context,
                                 lcmt_hsr_sim_status* output) const {
  DRAKE_DEMAND(output != nullptr);
  lcmt_hsr_sim_status& sim_status = *output;

  const auto& input = this->get_estimated_state_input_port().Eval(context);

  const int num_positions = robot_plant_->num_positions();
  for (drake::multibody::JointActuatorIndex i(0);
       i < robot_plant_->num_actuators(); ++i) {
    const auto& joint_i = robot_plant_->get_joint_actuator(i).joint();
    sim_status.joint_name.push_back(joint_i.name());
    sim_status.joint_position.push_back(input[joint_i.position_start()]);
    sim_status.joint_velocity.push_back(
        input[joint_i.velocity_start() + num_positions]);
    // Set the torque status to 0 for now.
    sim_status.joint_torque.push_back(0.0);
  }
}

}  // namespace hsr
}  // namespace examples
}  // namespace drake