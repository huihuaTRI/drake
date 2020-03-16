#include "drake/multibody/tree/door_hinge.h"

#include <memory>

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {

class DoorHingeTester {
 public:
  // Input argument door_hinge is aliased and must be valid whenever this
  // class exists.
  explicit DoorHingeTester(const DoorHinge<double>& door_hinge)
      : door_hinge_(door_hinge) {}

  double CalcHingeFrictionalTorque(double angular_velocity,
                                   const DoorHingeConfig& config) const {
    return door_hinge_.CalcHingeFrictionalTorque(angular_velocity, config);
  }

  double CalcHingeSpringTorque(double angle,
                               const DoorHingeConfig& config) const {
    return door_hinge_.CalcHingeSpringTorque(angle, config);
  }

  const DoorHingeConfig& door_hinge_config() const {
    return door_hinge_.config_;
  }

 private:
  const DoorHinge<double>& door_hinge_;
};

namespace {

constexpr double kPositionLowerLimit = -1.0;
constexpr double kPositionUpperLimit = 1.0;
constexpr double kDamping = 1.0;
constexpr double kAngle = 0.5;
constexpr double kAngularRate = 1.0;
constexpr double kIntegrationTimeStep = 1e-6;

class DoorHingeTest : public ::testing::Test {
 protected:
  // Based on the DoorHingeConfig to set up the door hinge joint and the plant.
  const DoorHingeConfig& BuildDoorHingeTester(const DoorHingeConfig& config) {
    systems::DiagramBuilder<double> builder;
    std::tie(plant_, scene_graph_) = AddMultibodyPlantSceneGraph(
        &builder, std::make_unique<MultibodyPlant<double>>(0.001));

    // Config the plant.
    benchmarks::inclined_plane::AddInclinedPlaneAndGravityToPlant(
        0.0 /*Set gravity to zero to simplify energy calculation*/,
        0.0 /*no inclination*/,
        std::nullopt /* Default inclined plan dimension*/,
        CoulombFriction(0.0, 0.0) /*no friction on the ground*/, plant_);

    // Neither the mass nor the center of mass position will affect the purpose
    // of the tests. They are set arbitrarily. The rotational inertia is also
    // set arbitrarily.
    const double kMass = 5.0;
    const Eigen::Vector3d kCoM = Eigen::Vector3d(0.1, 0.1, 0.2);
    auto& body1 = plant_->AddRigidBody(
        "body1", SpatialInertia<double>::MakeFromCentralInertia(
                     kMass, kCoM, RotationalInertia<double>(1.0, 1.0, 1.0)));
    plant_->RegisterVisualGeometry(body1, math::RigidTransformd::Identity(),
                                   geometry::Box{0.1, 0.1, 0.4}, "visual");
    plant_->RegisterCollisionGeometry(body1, math::RigidTransformd::Identity(),
                                      geometry::Box{0.1, 0.1, 0.4}, "collision",
                                      CoulombFriction(0.0, 0.0));

    revolute_joint_ = &plant_->AddJoint<RevoluteJoint>(
        "Joint1", plant_->world_body(), std::nullopt, body1, std::nullopt,
        Eigen::Vector3d::UnitZ(), kDamping);

    door_hinge_ = &plant_->AddForceElement<DoorHinge>(*revolute_joint_, config);

    // Finish building the model and create the context.
    plant_->Finalize();
    std::cout << "number of joints: " << plant_->num_joints() << std::endl;
    std::cout << "good here" << std::endl;

    // Add visualization for verification of the results when we have the
    // visualizer running.
    ConnectDrakeVisualizer(&builder, *scene_graph_, &lcm_);
    ConnectContactResultsToDrakeVisualizer(&builder, *plant_, &lcm_);
    diagram_ = builder.Build();
    std::cout << "good here" << std::endl;

    // Create a context for this system:
    diagram_context_ = diagram_->CreateDefaultContext();
    plant_context_ =
        &diagram_->GetMutableSubsystemContext(*plant_, diagram_context_.get());

    // // Create a tester for testing purpose.
    // door_hinge_tester_ = std::make_unique<DoorHingeTester>(*door_hinge_);

    // return *door_hinge_tester_;

    return config;
  }

  // void SetHingeJointState(double angle, double angular_rate) {
  //   revolute_joint_->set_angle(plant_context_, angle);
  //   revolute_joint_->set_angular_rate(plant_context_, angular_rate);
  // }

  const MultibodyPlant<double>& plant() const { return *plant_; }

  lcm::DrakeLcm lcm_;  // For visualization.
  MultibodyPlant<double>* plant_{nullptr};
  systems::Context<double>* plant_context_;
  geometry::SceneGraph<double>* scene_graph_{nullptr};
  std::unique_ptr<systems::Diagram<double>> diagram_;
  std::unique_ptr<systems::Context<double>> diagram_context_;
  const RevoluteJoint<double>* revolute_joint_{nullptr};
  const DoorHinge<double>* door_hinge_{nullptr};

  std::unique_ptr<DoorHingeTester> door_hinge_tester_;
};

DoorHingeConfig no_forces_config() {
  DoorHingeConfig config;
  config.spring_zero_angle_rad = 0;
  config.spring_constant = 0;
  config.dynamic_friction_torque = 0;
  config.static_friction_torque = 0;
  config.viscous_friction = 0;
  config.catch_width = 0;
  config.catch_torque = 0;
  return config;
}

// // With the condition that there is only frictional torque, this function
// // confirms that a) the potential energy and conservative power should be
// zero;
// // b) on-conservative power equals to the corresponding torque times
// velocity. void TestFrictionOnlyEnergyAndPower(const DoorHingeTester& dut,
//                                     const systems::Context<double>& context,
//                                     const MultibodyPlant<double>& plant,
//                                     double angular_rate) {
//   const double potential_energy_half_qc =
//   dut.door_hinge().CalcPotentialEnergy(
//       context, plant.EvalPositionKinematics(context));
//   EXPECT_EQ(potential_energy_half_qc, 0.0);

//   const double conserv_power = dut.door_hinge().CalcConservativePower(
//       context, plant.EvalPositionKinematics(context),
//       plant.EvalVelocityKinematics(context));
//   EXPECT_EQ(conserv_power, 0.0);

//   // Verify the non-conservative power
//   const double non_conserv_power = dut.door_hinge().CalcNonConservativePower(
//       context, plant.EvalPositionKinematics(context),
//       plant.EvalVelocityKinematics(context));
//   EXPECT_EQ(non_conserv_power, dut.CalcHingeFrictionalTorque(
//                                    angular_rate, dut.door_hinge_config()) *
//                                    angular_rate);
// }

// // With the condition that there is only spring related torque, this function
// // confirms that a) the potential energy equals to the corresponding torque
// // times velocity; b) on-conservative power should be zero.
// void TestSpringTorqueOnlyPower(const DoorHingeTester& dut,
//                                const systems::Context<double>& context,
//                                const MultibodyPlant<double>& plant,
//                                double angle, double angular_rate) {
//   auto spring_power = [&dut](double q, double v) {
//     return dut.CalcHingeSpringTorque(q, dut.door_hinge_config()) * v;
//   };

//   const double conserv_power = dut.door_hinge().CalcConservativePower(
//       context, plant.EvalPositionKinematics(context),
//       plant.EvalVelocityKinematics(context));
//   EXPECT_EQ(conserv_power, spring_power(angle, angular_rate));

//   const double non_conserv_power = dut.door_hinge().CalcNonConservativePower(
//       context, plant.EvalPositionKinematics(context),
//       plant.EvalVelocityKinematics(context));
//   EXPECT_EQ(non_conserv_power, 0.0);
// }

// // This function integrates the conservative power (P) to get the
// corresponding
// // energy (PE), i.e., PE = -∫Pdt. We assume the hinge joint moves from zero
// to
// // the input `angle` with a constant `angular_rate`.
// double IntegrateConservativePower(const DoorHingeTester& dut,
//                                   double target_angle, double angular_rate) {
//   auto conserv_power = [&dut](double q, double v) {
//     return dut.CalcHingeSpringTorque(q, dut.door_hinge_config()) * v;
//   };

//   double angle = 0.0;
//   double pe_integrated = 0.0;
//   while (angle < target_angle) {
//     pe_integrated -= conserv_power(angle, angular_rate) *
//     kIntegrationTimeStep; angle += angular_rate * kIntegrationTimeStep;
//   }
//   return pe_integrated;
// }

// Verify the torques and the energy should be zero when the config parameters
// are all zero.
TEST_F(DoorHingeTest, ZeroTest) {
  DoorHingeConfig config = no_forces_config();
  BuildDoorHingeTester(config);


  // // If no frictions, springs, etc. are applied, our torques should be 0.
  // EXPECT_EQ(dut.CalcHingeFrictionalTorque(0., config), 0);
  // EXPECT_EQ(dut.CalcHingeFrictionalTorque(1., config), 0);
  // EXPECT_EQ(dut.CalcHingeSpringTorque(0., config), 0);
  // EXPECT_EQ(dut.CalcHingeSpringTorque(1., config), 0);

  // // Test energy should be zero at a default condition.
  // const double potential_energy_1 = door_hinge_->CalcPotentialEnergy(
  //     *plant_context_, plant().EvalPositionKinematics(*plant_context_));
  // EXPECT_EQ(potential_energy_1, 0.0);
  // // Test energy should be zero at a non-default condition.
  // SetHingeJointState(0.2, 0.1);
  // const double potential_energy_2 = door_hinge_->CalcPotentialEnergy(
  //     *plant_context_, plant().EvalPositionKinematics(*plant_context_));
  // EXPECT_EQ(potential_energy_2, 0.0);
}

// // Test the case with only the torional spring torque, the corresponding
// energy
// // and power are computed correctly at different states.
// TEST_F(DoorHingeTest, SpringTest) {
//   DoorHingeConfig config = no_forces_config();
//   config.spring_constant = 1;
//   BuildDoorHingeTester(config);
//   DoorHingeTester dut = door_hinge_tester();

//   // Springs make spring torque (but not friction).
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(0., config), 0);
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(1., config), 0);
//   EXPECT_LT(dut.CalcHingeSpringTorque(1., config), 0);  // Pulls toward zero.
//   EXPECT_EQ(dut.CalcHingeSpringTorque(0., config), 0);

//   // Test potential energy at non-zero angle.
//   SetHingeJointState(kAngle, kAngularRate);
//   const double integrated_potential_energy =
//       IntegrateConservativePower(dut, kAngle, kAngularRate);
//   const double potential_energy = dut.door_hinge().CalcPotentialEnergy(
//       *context_, plant().EvalPositionKinematics(*context_));
//   EXPECT_NEAR(potential_energy, integrated_potential_energy,
//               kIntegrationTimeStep);

//   // Test the powers are computed correctly.
//   TestSpringTorqueOnlyPower(dut, *context_, plant(), kAngle, kAngularRate);
// }

// // Test the case with only the catch spring torque, the corresponding
// // energy and power are computed correctly at different states.
// TEST_F(DoorHingeTest, CatchTest) {
//   DoorHingeConfig config = no_forces_config();
//   config.catch_width = 2 * kAngle;
//   config.catch_torque = 1.0;
//   BuildDoorHingeTester(config);
//   DoorHingeTester dut = door_hinge_tester();

//   // The catch makes spring torque (but not friction).
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(0., config), 0);
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(1., config), 0);

//   // Resists closing.
//   EXPECT_GT(dut.CalcHingeSpringTorque(config.catch_width, config), 0);
//   // Tipover point.
//   EXPECT_EQ(dut.CalcHingeSpringTorque(config.catch_width / 2, config), 0);
//   // Detent pulls in.
//   EXPECT_LT(dut.CalcHingeSpringTorque(0., config), 0);

//   // Verify that the potential energy with only the catch spring torque at
//   zero
//   // position and the catch_width position should be 0.
//   const double potential_energy_q0 = dut.door_hinge().CalcPotentialEnergy(
//       *context_, plant().EvalPositionKinematics(*context_));
//   EXPECT_EQ(potential_energy_q0, 0.0);

//   SetHingeJointState(config.catch_width, 0.0);
//   const double potential_energy_qc = dut.door_hinge().CalcPotentialEnergy(
//       *context_, plant().EvalPositionKinematics(*context_));
//   EXPECT_EQ(potential_energy_qc, 0.0);

//   // Test the energy from power integration.
//   SetHingeJointState(kAngle, kAngularRate);
//   const double integrated_potential_energy =
//       IntegrateConservativePower(dut, kAngle, kAngularRate);
//   const double potential_energy = dut.door_hinge().CalcPotentialEnergy(
//       *context_, plant().EvalPositionKinematics(*context_));
//   EXPECT_NEAR(potential_energy, integrated_potential_energy,
//               kIntegrationTimeStep);

//   // Verify the power terms are computed correctly.
//   // Test the powers are computed correctly
//   TestSpringTorqueOnlyPower(dut, *context_, plant(), kAngle, kAngularRate);
// }

// TEST_F(DoorHingeTest, StaticFrictionTest) {
//   DoorHingeConfig config = no_forces_config();
//   config.static_friction_torque = 1;
//   BuildDoorHingeTester(config);
//   DoorHingeTester dut = door_hinge_tester();

//   // Friction opposes tiny motion, but falls away with substantial motion.
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(0., config), 0);
//   EXPECT_LT(dut.CalcHingeFrictionalTorque(0.001, config), -0.5);
//   EXPECT_GT(dut.CalcHingeFrictionalTorque(-0.001, config), 0.5);
//   EXPECT_NEAR(dut.CalcHingeFrictionalTorque(0.01, config), 0, 1e-7);

//   // No spring torque.
//   EXPECT_EQ(dut.CalcHingeSpringTorque(0., config), 0);
//   EXPECT_EQ(dut.CalcHingeSpringTorque(1., config), 0);

//   // Test the energy and power with a given state.
//   SetHingeJointState(kAngle, kAngularRate);
//   TestFrictionOnlyEnergyAndPower(dut, *context_, plant(), kAngularRate);
// }

// TEST_F(DoorHingeTest, DynamicFrictionTest) {
//   DoorHingeConfig config = no_forces_config();
//   config.dynamic_friction_torque = 1;
//   BuildDoorHingeTester(config);
//   DoorHingeTester dut = door_hinge_tester();

//   // Friction opposes any motion, even tiny motion.
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(0., config), 0);
//   EXPECT_LT(dut.CalcHingeFrictionalTorque(0.001, config), -0.5);
//   EXPECT_GT(dut.CalcHingeFrictionalTorque(-0.001, config), 0.5);
//   EXPECT_NEAR(dut.CalcHingeFrictionalTorque(0.01, config), -1, 1e-7);

//   // No spring torque.
//   EXPECT_EQ(dut.CalcHingeSpringTorque(0., config), 0);
//   EXPECT_EQ(dut.CalcHingeSpringTorque(1., config), 0);

//   // Test the energy and power with a given state.
//   SetHingeJointState(kAngle, kAngularRate);
//   TestFrictionOnlyEnergyAndPower(dut, *context_, plant(), kAngularRate);
// }

// TEST_F(DoorHingeTest, ViscousFrictionTest) {
//   DoorHingeConfig config = no_forces_config();
//   config.viscous_friction = 1;
//   BuildDoorHingeTester(config);
//   DoorHingeTester dut = door_hinge_tester();

//   // Friction opposes motion proprotionally.
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(0., config), 0);
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(1., config), -1);
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(-1., config), 1);
//   EXPECT_EQ(dut.CalcHingeFrictionalTorque(-2., config), 2);

//   // No spring torque.
//   EXPECT_EQ(dut.CalcHingeSpringTorque(0., config), 0);
//   EXPECT_EQ(dut.CalcHingeSpringTorque(1., config), 0);

//   // Test the energy and power with a given state.
//   SetHingeJointState(kAngle, kAngularRate);
//   TestFrictionOnlyEnergyAndPower(dut, *context_, plant(), kAngularRate);
// }

// TEST_F(DoorHingeTest, EnergyConservationTest) {
//   // Use the default door hinge configuration.
//   DoorHingeConfig config{};
//   BuildDoorHingeTester(config);

//   systems::Simulator<double> simulator(*plant_);

//   auto& plant_contest = plant_->GetMutableSubsystemContext(
//       *plant_, &simulator.get_mutable_context());
//   revolute_joint_->set_angle(&plant_contest, 0.0);
//   revolute_joint_->set_angular_rate(&plant_contest, 0.1);

//   const double kCheckValueStepSize = 0.1;
//   const double kTotalSimTime = 1.0;
//   double sim_time = kCheckValueStepSize;
//   while (sim_time <= kTotalSimTime) {
//     simulator.AdvanceTo(sim_time);
//     sim_time += kCheckValueStepSize;

//     auto& plant_contest_temp =
//         plant_->GetSubsystemContext(*plant_, simulator.get_context());
//     const Eigen::VectorXd x = plant_->GetPositions(plant_contest_temp);
//     std::cout << "Plant state: " << x << std::endl;
//     std::cout << "Potential energy: "
//               << plant_->CalcPotentialEnergy(plant_contest_temp) <<
//               std::endl;
//   }
// }

}  // namespace
}  // namespace multibody
}  // namespace drake
