#include "drake/examples/pr2/main_controller.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/examples/pr2/robot_parameters_loader.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/parsing/parser.h"

namespace drake {
namespace examples {
namespace pr2 {
namespace {

constexpr double kMaxTimeStep = 1e-3;

class MainControllerTest : public ::testing::Test {
 public:
  MainControllerTest() {
    LoadPlant();

    LoadRobotParameters();

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

  void LoadRobotParameters() {
    const bool load_successful = ReadParametersFromFile(
        robot_name_, filepath_prefix_, &robot_parameters_);
    DRAKE_DEMAND(load_successful);
  }

  void PopulateTestingStatesValue() {
    const int num_positions = plant_.num_positions();
    const int num_velocities = plant_.num_velocities();
    const int state_size = num_positions + num_velocities;

    estimated_state_ = Eigen::VectorXd::Zero(state_size);
    estimated_state_[0] = 1;
    desired_state_ = Eigen::VectorXd::Zero(state_size);
    desired_state_[0] = 1;

    for (int i = 4; i < state_size; ++i) {
      estimated_state_[i] = i * i;
      desired_state_[i] = i * i * i;
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

TEST_F(MainControllerTest, ConstructionTest) {
  MainController controller_test(plant(), welded_plant(), robot_parameters());

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
  // non zero value and the actuation port output should always be zero.
  // Since the first three elements are always 0 (corresponding to the
  // floating base), we start from the forth one.
  const double kErrorTol = 0.01;

  int non_zero_count = 0;
  for (int i = 3; i < plant().num_velocities(); ++i) {
    if (std::abs(output_generalized_force[i]) > kErrorTol) {
      non_zero_count += 1;
    }
  }
  EXPECT_GT(non_zero_count, 1);
}

}  // namespace
}  // namespace pr2
}  // namespace examples
}  // namespace drake
