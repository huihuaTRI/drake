#include "drake/examples/pr2/sim_world.h"

#include <unordered_set>
#include <utility>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/pr2/pr2_pd_controller.h"
#include "drake/examples/pr2/pr2_upper_body_controller.h"
#include "drake/examples/pr2/robot_parameters.h"
#include "drake/geometry/render/render_engine_vtk_factory.h"
#include "drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/systems/primitives/pass_through.h"

namespace drake {
namespace examples {
namespace pr2 {

using drake::geometry::SceneGraph;
using drake::geometry::render::MakeRenderEngineVtk;
using drake::geometry::render::RenderEngineVtkParams;
using drake::math::RigidTransform;
using drake::multibody::BodyIndex;
using drake::multibody::Frame;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::Context;
using drake::systems::Diagram;
using drake::systems::State;

namespace {
constexpr double kGravity = 9.8;
constexpr double kPenetrationAllowance = 0.6;
constexpr double kVStictionTolerance = 0.6;
constexpr double kInclinedPlaneStaticFrictionCoef = 0.6;
constexpr double kInclinedPlaneKineticFrictionCoef = 0.6;
}  // namespace

template <typename T>
SimWorld<T>::SimWorld(const std::string& robot_name, double time_step)
    : robot_name_(robot_name),
      owned_plant_(std::make_unique<MultibodyPlant<T>>(time_step)),
      owned_scene_graph_(std::make_unique<SceneGraph<T>>()),
      owned_robot_plant_(std::make_unique<MultibodyPlant<T>>(time_step)),
      owned_welded_robot_plant_(
          std::make_unique<MultibodyPlant<T>>(time_step)) {
  DRAKE_DEMAND(!robot_name.empty());
  DRAKE_DEMAND(time_step > 0);
  // This class holds the unique_ptrs explicitly for plant and scene_graph
  // until Finalize() is called (when they are moved into the Diagram). Grab
  // the raw pointers, which should stay valid for the lifetime of the Diagram.
  scene_graph_ = owned_scene_graph_.get();
  scene_graph_->set_name("scene_graph");
  // Setup the render engine. Choose to use the default for now.
  scene_graph_->AddRenderer("sim_world_renderer",
                            MakeRenderEngineVtk(RenderEngineVtkParams()));

  plant_ = owned_plant_.get();
  plant_->RegisterAsSourceForSceneGraph(scene_graph_);
  plant_->set_name("sim_world_plant");

  LoadModelsAndSetSimWorld();

  Finalize();

  // Set the diagram name.
  this->set_name("sim_world");
}

template <typename T>
void SimWorld<T>::LoadModelsAndSetSimWorld() {
  // Add a ground plane.
  {
    const multibody::CoulombFriction<double> coef_friction_inclined_plane(
        kInclinedPlaneStaticFrictionCoef, kInclinedPlaneKineticFrictionCoef);
    multibody::benchmarks::inclined_plane::AddInclinedPlaneAndGravityToPlant(
        kGravity, 0.0, std::nullopt, coef_friction_inclined_plane, plant_);
  }

  // Add a robot model.
  {
    const std::string filepath_prefix = "drake/examples/pr2/config/";
    DRAKE_DEMAND(ReadParametersFromFile(robot_name_, filepath_prefix,
                                        &robot_parameters_));
    DRAKE_DEMAND(robot_parameters_.name == robot_name_);

    const std::string model_path =
        FindResourceOrThrow(robot_parameters_.model_instance_info.model_path);
    Parser parser(plant_);
    const multibody::ModelInstanceIndex model_instance_index =
        parser.AddModelFromFile(model_path, robot_name_);

    // Filling the rest of the model instance information.
    robot_parameters_.model_instance_info.model_path = model_path;
    robot_parameters_.model_instance_info.index = model_instance_index;
    robot_parameters_.model_instance_info.X_PC =
        RigidTransform<double>::Identity();
  }

  // Other environmental models can be added here.
}

template <typename T>
void SimWorld<T>::SetDefaultState(const Context<T>& context,
                                  State<T>* state) const {
  // Call the base class method, to initialize all systems in this diagram.
  Diagram<T>::SetDefaultState(context, state);

  const auto& plant_context = this->GetSubsystemContext(*plant_, context);

  drake::log()->info("Setting initial position of robot: " + robot_name_);
  const auto& robot_instance_index =
      robot_parameters_.model_instance_info.index;
  SetModelPositionState(
      context, robot_instance_index,
      plant_->GetPositions(plant_context, robot_instance_index), state);
  SetModelVelocityState(
      context, robot_instance_index,
      plant_->GetVelocities(plant_context, robot_instance_index), state);
}

template <typename T>
void SimWorld<T>::SetModelPositionState(
    const Context<T>& context, const ModelInstanceIndex& model_index,
    const Eigen::Ref<const drake::VectorX<T>>& q, State<T>* state) const {
  DRAKE_DEMAND(state != nullptr);
  const int num_model_positions = plant_->num_positions(model_index);
  DRAKE_DEMAND(q.size() == num_model_positions);
  auto& plant_context = this->GetSubsystemContext(*plant_, context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetPositions(plant_context, &plant_state, model_index, q);
}

template <typename T>
void SimWorld<T>::SetModelVelocityState(
    const Context<T>& context, const ModelInstanceIndex& model_index,
    const Eigen::Ref<const drake::VectorX<T>>& v, State<T>* state) const {
  DRAKE_DEMAND(state != nullptr);
  const int num_model_velocities = plant_->num_velocities(model_index);
  DRAKE_DEMAND(v.size() == num_model_velocities);
  auto& plant_context = this->GetSubsystemContext(*plant_, context);
  auto& plant_state = this->GetMutableSubsystemState(*plant_, state);
  plant_->SetVelocities(plant_context, &plant_state, model_index, v);
}

template <typename T>
void SimWorld<T>::MakeRobotControlPlants() {
  // Build the fmk plants for controller purpose. It contains both the floating
  // FMK robot model and the same model but with the base welded to the ground.
  const auto& model_info = robot_parameters_.model_instance_info;
  Parser(owned_robot_plant_.get()).AddModelFromFile(model_info.model_path);
  owned_robot_plant_->set_name("robot_plant");
  owned_robot_plant_->Finalize();

  // Create the welded version for inverse dynamic controller.
  const auto welded_robot_model = Parser(owned_welded_robot_plant_.get())
                                      .AddModelFromFile(model_info.model_path);

  // The welded plant is only used for the inverse dynamics controller
  // calculation purpose. Weld the plant only it's a floating base robot.
  if (model_info.is_floating_base) {
    owned_welded_robot_plant_->WeldFrames(
        owned_welded_robot_plant_->world_frame(),
        owned_welded_robot_plant_->GetFrameByName(model_info.child_frame_name,
                                                  welded_robot_model),
        model_info.X_PC);
  }
  owned_welded_robot_plant_->set_name("welded_" + robot_name_);
  owned_welded_robot_plant_->Finalize();
}

template <typename T>
void SimWorld<T>::Finalize() {
  MakeRobotControlPlants();

  // Note: This deferred diagram construction method/workflow exists because we
  //   - cannot finalize plant until all of the objects are added, and
  //   - cannot wire up the diagram until we have finalized the plant.
  plant_->Finalize();
  plant_->set_penetration_allowance(kPenetrationAllowance);
  plant_->set_stiction_tolerance(kVStictionTolerance);

  systems::DiagramBuilder<T> builder;
  builder.AddSystem(std::move(owned_plant_));
  builder.AddSystem(std::move(owned_scene_graph_));

  builder.Connect(
      plant_->get_geometry_poses_output_port(),
      scene_graph_->get_source_pose_port(plant_->get_source_id().value()));
  builder.Connect(scene_graph_->get_query_output_port(),
                  plant_->get_geometry_query_input_port());

  const auto& robot_instance = robot_parameters_.model_instance_info.index;
  const int num_robot_positions = plant_->num_positions(robot_instance);
  const int num_robot_velocities = plant_->num_velocities(robot_instance);
  const int robot_state_size = num_robot_positions + num_robot_velocities;

  // Export Robot "state" outputs.
  builder.ExportOutput(plant_->get_state_output_port(robot_instance),
                       robot_name_ + "_estimated_state");

  // Connect the states with controllers.
  {
    // Creates desired state and estimated state passthroughs to connect to
    // different controllers.
    auto& desired_state_passthrough =
        *builder.template AddSystem<drake::systems::PassThrough<double>>(
            robot_state_size);
    // Exposes the desired state port.
    builder.ExportInput(desired_state_passthrough.get_input_port(),
                        robot_name_ + "_desired_state");

    auto& estimated_state_passthrough =
        *builder.template AddSystem<drake::systems::PassThrough<double>>(
            robot_state_size);
    builder.Connect(plant_->get_state_output_port(robot_instance),
                    estimated_state_passthrough.get_input_port());

    // Create the upper body controller and connect the states.
    auto& upper_body_controller =
        *builder.template AddSystem<Pr2UpperBodyController>(
            *owned_robot_plant_, *owned_welded_robot_plant_, robot_parameters_);
    builder.Connect(desired_state_passthrough.get_output_port(),
                    upper_body_controller.get_desired_state_input_port());
    builder.Connect(estimated_state_passthrough.get_output_port(),
                    upper_body_controller.get_estimated_state_input_port());

    // Create the chassis controller and connect the states.
    const std::string kChassisPartName = "chassis";
    const auto chassis_parameters =
        robot_parameters_.parts_parameters.find(kChassisPartName);
    DRAKE_DEMAND(chassis_parameters !=
                 robot_parameters_.parts_parameters.end());
    auto& chassis_pd_controller = *builder.template AddSystem<Pr2PdController>(
        *owned_robot_plant_, chassis_parameters->second);
    builder.Connect(desired_state_passthrough.get_output_port(),
                    chassis_pd_controller.get_desired_state_input_port());
    builder.Connect(estimated_state_passthrough.get_output_port(),
                    chassis_pd_controller.get_estimated_state_input_port());

    // Create two selector matrix to pick the corresponding generalized forces
    // for each of the controller.
    Eigen::MatrixXd upper_body_selector =
        Eigen::MatrixXd::Zero(num_robot_velocities, num_robot_velocities);
    // Initially, we set all the joints to use upper body inverse dynamics
    // controller.
    const int num_robot_actuators = owned_robot_plant_->num_actuators();
    upper_body_selector.bottomRightCorner(num_robot_actuators,
                                          num_robot_actuators) =
        Eigen::MatrixXd::Identity(num_robot_actuators, num_robot_actuators);

    Eigen::MatrixXd chassis_selector =
        Eigen::MatrixXd::Zero(num_robot_velocities, num_robot_velocities);
    // Update the selector matrix.
    for (const auto& joint_control_info :
         chassis_pd_controller.part_control_info()) {
      const int joint_velocity_index = joint_control_info.velocity_index -
                                       owned_robot_plant_->num_positions();
      chassis_selector(joint_velocity_index, joint_velocity_index) = 1;
      upper_body_selector(joint_velocity_index, joint_velocity_index) = 0;
    }

    // Create MatrixGain systems for the selectors.
    auto& chassis_control_selector_system =
        *builder.template AddSystem<drake::systems::MatrixGain<double>>(
            chassis_selector);
    builder.Connect(chassis_pd_controller.get_generalized_force_output_port(),
                    chassis_control_selector_system.get_input_port());

    // Create the selector system of the upper body controller.
    auto& upper_body_control_selector_system =
        *builder.template AddSystem<drake::systems::MatrixGain<double>>(
            upper_body_selector);
    builder.Connect(upper_body_controller.get_generalized_force_output_port(),
                    upper_body_control_selector_system.get_input_port());

    // Create an adder to add these two controllers together.
    const int kNumControllers = 2;
    auto& adder = *builder.template AddSystem<drake::systems::Adder<double>>(
        kNumControllers, num_robot_velocities);
    builder.Connect(chassis_control_selector_system.get_output_port(),
                    adder.get_input_port(0));
    builder.Connect(upper_body_control_selector_system.get_output_port(),
                    adder.get_input_port(1));

    // Finally, connect the adder to the plant input.
    builder.Connect(adder.get_output_port(),
                    plant_->get_applied_generalized_force_input_port());

    VectorX<double> constant_actuation_value =
        VectorX<double>::Zero(plant_->num_actuators());
    auto& actuation_constant_source =
        *builder.template AddSystem<systems::ConstantVectorSource<double>>(
            constant_actuation_value);

    builder.Connect(actuation_constant_source.get_output_port(),
                    plant_->get_actuation_input_port(robot_instance));
  }

  builder.ExportOutput(scene_graph_->get_pose_bundle_output_port(),
                       "pose_bundle");
  builder.ExportOutput(plant_->get_contact_results_output_port(),
                       "contact_results");
  builder.ExportOutput(plant_->get_geometry_poses_output_port(),
                       "geometry_poses");

  builder.BuildInto(this);
}

}  // namespace pr2
}  // namespace examples
}  // namespace drake

template class drake::examples::pr2::SimWorld<double>;
