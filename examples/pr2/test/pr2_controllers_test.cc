#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/examples/pr2/pr2_pd_controller.h"
#include "drake/examples/pr2/pr2_upper_body_controller.h"
#include "drake/examples/pr2/robot_parameters.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/parsing/parser.h"

namespace drake {
namespace examples {
namespace pr2 {
namespace {

constexpr double kMaxTimeStep = 1e-3;

class Pr2ControllersTest : public ::testing::Test {
 public:
  Pr2ControllersTest() {
    LoadPlant();

    // Load the PR2 parameters.
    const bool load_successful = ReadParametersFromFile(
        robot_name_, filepath_prefix_, &robot_parameters_);
    DRAKE_DEMAND(load_successful);

    PopulateTestingStatesValue();
  }

  const Eigen::VectorXd& estimated_state() { return estimated_state_; }
  const Eigen::VectorXd& desired_state() { return desired_state_; }

  const multibody::MultibodyPlant<double>& plant() { return plant_; }
  const multibody::MultibodyPlant<double>& welded_plant() {
    return welded_plant_;
  }

  const RobotParameters& robot_parameters() { return robot_parameters_; }

  void LoadPlant() {
    const std::string model_path = FindResourceOrThrow(model_path_);
    multibody::Parser(&plant_).AddModelFromFile(model_path);
    plant_.Finalize();

    // Create the same plant but with the base welded to the ground.
    // The welded plant is only used for the inverse dynamics controller
    // calculation purpose. Here we assume the robot only has one floating
    // body, which should be true.
    multibody::Parser(&welded_plant_).AddModelFromFile(model_path);
    welded_plant_.Finalize();
  }

  void PopulateTestingStatesValue() {
    const int num_positions = plant_.num_positions();
    const int num_velocities = plant_.num_velocities();
    const int state_size = num_positions + num_velocities;

    estimated_state_ = Eigen::VectorXd::Zero(state_size);
    desired_state_ = Eigen::VectorXd::Zero(state_size);

    for (int i = 0; i < state_size; ++i) {
      estimated_state_[i] = 0.1 * i;
      desired_state_[i] = 0.2 * i;
    }
  }

 private:
  const std::string robot_name_ = "pr2";
  const std::string filepath_prefix_ = "drake/examples/pr2/config/";
  const std::string model_path_ =
      "drake/examples/pr2/models/pr2_description/urdf/pr2_simplified.urdf";
  multibody::MultibodyPlant<double> plant_{kMaxTimeStep};
  multibody::MultibodyPlant<double> welded_plant_{kMaxTimeStep};

  RobotParameters robot_parameters_;

  Eigen::VectorXd estimated_state_;
  Eigen::VectorXd desired_state_;
};

TEST_F(Pr2ControllersTest, UpperBodyControllerTest) {
  Pr2UpperBodyController controller_test(plant(), welded_plant(),
                                         robot_parameters());

  std::unique_ptr<systems::Context<double>> context =
      controller_test.CreateDefaultContext();

  controller_test.get_desired_state_input_port().FixValue(context.get(),
                                                          desired_state());
  controller_test.get_estimated_state_input_port().FixValue(context.get(),
                                                            estimated_state());

  const auto& output_generalized_force =
      controller_test.get_generalized_force_output_port().Eval(*context);

  // It's hard to predict the values coming out from the Drake inverse dynamics
  // controller. Here, we check that at least two generalized forces should be
  // non zero value.
  const double kErrorTol = 0.01;

  int non_zero_count = 0;
  for (int i = 3; i < plant().num_velocities(); ++i) {
    if (std::abs(output_generalized_force[i]) > kErrorTol) {
      non_zero_count += 1;
    }
  }
  EXPECT_GT(non_zero_count, 1);
}

// Tests the nominal case that the input ports are connected properly.
TEST_F(Pr2ControllersTest, PdControllerTest) {
  const std::string kTestPartName = "chassis";
  const auto chassis_parameters =
      robot_parameters().parts_parameters.find(kTestPartName);
  DRAKE_DEMAND(chassis_parameters != robot_parameters().parts_parameters.end());
  Pr2PdController controller_test(plant(), chassis_parameters->second);
  std::unique_ptr<drake::systems::Context<double>> context =
      controller_test.CreateDefaultContext();

  // Sets the input ports.
  controller_test.get_desired_state_input_port().FixValue(context.get(),
                                                          desired_state());
  controller_test.get_estimated_state_input_port().FixValue(context.get(),
                                                            estimated_state());

  // Checks the output value.
  const auto& output =
      controller_test.get_generalized_force_output_port().Eval(*context);

  const auto& part_control_info = controller_test.part_control_info();
  int i = 0;
  int saturated_torque = 0;
  int not_saturated_torque = 0;
  for (const auto& joint_control_info : part_control_info) {
    const double pos_error =
        desired_state()[joint_control_info.position_index] -
        estimated_state()[joint_control_info.position_index];
    const double vel_error =
        desired_state()[joint_control_info.velocity_index] -
        estimated_state()[joint_control_info.velocity_index];
    const double torque =
        joint_control_info.kp * pos_error + joint_control_info.kd * vel_error;
    // If torque is saturated, the computed torque should reach the
    // maximum effort limit.
    if (std::abs(torque) > joint_control_info.effort_limit) {
      EXPECT_DOUBLE_EQ(std::abs(output[joint_control_info.velocity_index -
                                       plant().num_positions()]),
                       joint_control_info.effort_limit);
      ++saturated_torque;
    } else {
      EXPECT_DOUBLE_EQ(
          output[joint_control_info.velocity_index - plant().num_positions()],
          torque);
      ++not_saturated_torque;
    }
    ++i;
  }
}

}  // namespace
}  // namespace pr2
}  // namespace examples
}  // namespace drake
