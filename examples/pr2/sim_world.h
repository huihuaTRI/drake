#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/examples/pr2/robot_parameters.h"
#include "drake/geometry/render/render_engine.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace examples {
namespace pr2 {

/// A diagram system that represents the complete simulation world environment,
/// including the robots and anything a user might want to load into the model
/// such as objects or sensors.
///
/// @system{SimWorld,
///   @input_port{[name]_desired_state}
///   @output_port{[name]_estimated_state}
///   @output_port{pose_bundle}
///   @output_port{contact_results}
///   @output_port{geometry_poses}
/// }
/// The exact name of the port will depend on the input name of the robot.

template <typename T>
class SimWorld : public systems::Diagram<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SimWorld);
  /// Default constructor.
  /// @param robot_name  The name of the robot that will be loaded. The default
  ///    is `pr2`.
  /// @param time_step  Timestep of the multibody plant. It should be greater
  ///    than 0.
  explicit SimWorld(const std::string& robot_name = "pr2",
                    double time_step = 1e-3);

  /// Sets the default State using the Diagram's default state but override
  /// free body positions with data from the model directive yaml files and the
  /// default joint positions and velocities of the robot from
  /// GetModelPositionState() and GetModelVelocityState().
  /// @param context A const reference to the SimWorld context.
  /// @param state A pointer to the State of the SimWorld system.
  /// @pre `state` must be the systems::State<T> object contained in `context`.
  void SetDefaultState(const systems::Context<T>& context,
                       systems::State<T>* state) const override;

  /// Returns a reference to the internal robot plant used for the internal
  /// controller.
  const multibody::MultibodyPlant<T>& get_robot_plant() const {
    return *owned_robot_plant_;
  }

  /// Returns a reference to the main plant responsible for the dynamics of
  /// the robot and the environment.
  const multibody::MultibodyPlant<T>& get_sim_world_plant() const {
    return *plant_;
  }

  /// Returns a mutable reference to the main plant responsible for the
  /// dynamics of the robot and the environment. Mutation of this plant must be
  /// done before calling Finalize().
  multibody::MultibodyPlant<T>& get_mutable_sim_world_plant() {
    return *plant_;
  }

  /// Returns a mutable reference to the internal SceneGraph responsible for all
  /// of the geometry for the robot and the environment.
  drake::geometry::SceneGraph<T>& get_mutable_scene_graph() {
    return *scene_graph_;
  }

  /// Sets the position state of a model that has at least one free moving
  /// joint. For example, the model could be a robot or a dishwasher.
  /// @param context A const reference to the SimWorld context.
  /// @param model_index Const reference of the interested model instance index.
  /// @param q The target position state of the corresponding model to be set.
  /// @param state The full state of the robot world, must be the
  /// systems::State<T> object contained in `context`.
  void SetModelPositionState(const systems::Context<T>& context,
                             const multibody::ModelInstanceIndex& model_index,
                             const Eigen::Ref<const VectorX<T>>& q,
                             systems::State<T>* state) const;

  /// Sets the velocity state of a model that has at least one free moving
  /// joint. For example, the model could be a robot or a dishwasher.
  /// @param context A const reference to the SimWorld context.
  /// @param model_index Const reference of the interested model instance index.
  /// @param v The target velocity state of the corresponding model to be set.
  /// @param state The full state of the SimWorld, must be the
  /// systems::State<T> object contained in `context`.
  void SetModelVelocityState(const systems::Context<T>& context,
                             const multibody::ModelInstanceIndex& model_index,
                             const Eigen::Ref<const VectorX<T>>& v,
                             systems::State<T>* state) const;

 private:
  // Finalizes the multibody plant before using this class in the Systems
  // framework. This should be called exactly once. See
  // multibody::MultibodyPlant<T>::Finalize().
  void Finalize();

  // This function encapsulates all the model loading related actions. For
  // example, user could set up the ground plane, load all the models from URDFs
  // or SDFs and set up sensors (such as camera and force sensor), etc.
  void LoadModelsAndSetSimWorld();

  // Creates two models for controller purpose. One model is the original model
  // and one model is the welded version of the original model, independent
  // of whether the original model is welded or not. If the original model is
  // already a fixed-base robot, the original version and the welded version
  // will be the same. However, here we keep two models to make sure the logic
  // works for both fixed-base models and floating-base models. This function
  // assumes fmk_model_info_ has already being populated. Should only be called
  // from Finalize().
  void MakeRobotControlPlants();

  const std::string robot_name_;
  RobotParameters robot_parameters_;

  // These are only valid until Finalize() is called.
  std::unique_ptr<multibody::MultibodyPlant<T>> owned_plant_;
  std::unique_ptr<geometry::SceneGraph<T>> owned_scene_graph_;

  // These are valid for the lifetime of this system.
  multibody::MultibodyPlant<T>* plant_{};
  geometry::SceneGraph<T>* scene_graph_{};
  std::unique_ptr<drake::multibody::MultibodyPlant<T>> owned_robot_plant_;
  std::unique_ptr<drake::multibody::MultibodyPlant<T>>
      owned_welded_robot_plant_;
};

}  // namespace pr2
}  // namespace examples
}  // namespace drake
