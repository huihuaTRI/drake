/// Simulate a camera that views a coke can

#include <limits>
#include <memory>

#include <gflags/gflags.h>

#include "robotlocomotion/header_t.hpp"
#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/dev/render/render_engine.h"
#include "drake/geometry/dev/render/render_engine_vtk.h"
#include "drake/geometry/dev/scene_graph.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/sensors/dev/rgbd_camera.h"
#include "drake/systems/sensors/image_to_lcm_image_array_t.h"
#include "drake/systems/sensors/pixel_types.h"

namespace drake {
namespace examples {
namespace coke_cam {
namespace {

DEFINE_double(max_time_step, 3.0e-4,
              "Simulation time step used for integrator.");
DEFINE_double(realtime_rate, 1.0,
              "Rate at which to run the simulation, relative to realtime");
DEFINE_double(simulation_time, std::numeric_limits<double>::infinity(),
              "How long to simulate the pendulum");
DEFINE_double(inclined_plane_coef_static_friction, 0.3,
              "Inclined plane's coefficient of static friction (no units).");
DEFINE_double(inclined_plane_coef_kinetic_friction, 0.3,
              "Inclined plane's coefficient of kinetic friction (no units).  "
              "When time_step > 0, this value is ignored.  Only the "
              "coefficient of static friction is used in fixed-time step.");
DEFINE_double(gravity, 9.8, "Value of gravity in the direction of -z.");

// Urdf path of the coke can and a reference geometry.
static const char* const kCokeCanUrdfPath =
    "drake/examples/coke_cam/urdf/coke_can.urdf";
static const char* const kUnitBoxUrdfPath =
    "drake/examples/coke_cam/urdf/couch_rough.urdf";

int do_main() {
  DRAKE_DEMAND(FLAGS_simulation_time > 0.0);

  systems::DiagramBuilder<double> builder;

  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Create real model for simulation.
  multibody::MultibodyPlant<double>& plant =
      *builder.AddSystem<multibody::MultibodyPlant>(FLAGS_max_time_step);
  plant.set_name("plant");

  plant.RegisterAsSourceForSceneGraph(&scene_graph);

  multibody::Parser parser(&plant);

  const std::string can_full_name = FindResourceOrThrow(kCokeCanUrdfPath);

  auto can_model_instance_index = parser.AddModelFromFile(can_full_name);
  (void)can_model_instance_index;

  // Add a reference unit cubit.
  const std::string cube_full_name = FindResourceOrThrow(kUnitBoxUrdfPath);

  auto cube_model_instance_index = parser.AddModelFromFile(cube_full_name);
  (void)cube_model_instance_index;

  // Add half space plane and gravity.
  const drake::multibody::CoulombFriction<double> coef_friction_inclined_plane(
      FLAGS_inclined_plane_coef_static_friction,
      FLAGS_inclined_plane_coef_kinetic_friction);
  drake::multibody::benchmarks::inclined_plane::
      AddInclinedPlaneAndGravityToPlant(FLAGS_gravity, 0.0, drake::nullopt,
                                        coef_friction_inclined_plane, &plant);

  plant.Finalize();

  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  DRAKE_DEMAND(!!plant.get_source_id());
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  geometry::ConnectDrakeVisualizer(&builder, scene_graph);

  // Create scene render.
  const std::string default_render_name = "coke_cam_render";
  geometry::dev::SceneGraph<double>* render_scene_graph{};
  render_scene_graph =
      builder.template AddSystem<drake::geometry::dev::SceneGraph>();
  render_scene_graph->set_name("dev_scene_graph_for_rendering");
  render_scene_graph->AddRenderer(
      default_render_name,
      std::make_unique<drake::geometry::dev::render::RenderEngineVtk>());
  render_scene_graph->CopyFrom(scene_graph);

  builder.Connect(
      plant.get_geometry_poses_output_port(),
      render_scene_graph->get_source_pose_port(plant.get_source_id().value()));

  // Setup a camera.
  const std::string camera_name = "ground_cam";
  const double kFocalY = 645;
  const int kHeight = 480;
  const int kWidth = 848;
  const double fov_y = std::atan(kHeight / 2. / kFocalY) * 2;
  drake::geometry::dev::render::DepthCameraProperties camera_properties(
      kWidth, kHeight, fov_y, default_render_name, 0.1, 2.0);
  // Determine the camera to world frame orientation.
  drake::math::RotationMatrix<double> R_AB =
      drake::math::RotationMatrix<double>::MakeFromOrthonormalRows(
          Eigen::Vector3d{0, -1, 0}, Eigen::Vector3d{1, 0, 0},
          Eigen::Vector3d{0, 0, 1});
  // Rotate the camera to point downward slightly.
  drake::math::RotationMatrix<double> R_WA(
      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()));

  auto& cam_parent_frame = plant.world_body().body_frame();
  const drake::optional<drake::geometry::FrameId> parent_body_id =
      plant.GetBodyFrameIdIfExists(cam_parent_frame.body().index());
  DRAKE_THROW_UNLESS(parent_body_id.has_value());
  const drake::Isometry3<double> X_PC =
      cam_parent_frame.GetFixedPoseInBodyFrame() *
      math::RigidTransform<double>(R_WA * R_AB,
                                   Eigen::Vector3d(0.0, -1.0, 0.1));

  auto camera =
      builder.template AddSystem<drake::systems::sensors::dev::RgbdCamera>(
          camera_name, parent_body_id.value(), X_PC, camera_properties, false);
  builder.Connect(render_scene_graph->get_query_output_port(),
                  camera->query_object_input_port());

  // Publish the image.
  auto image_to_lcm_image_array =
      builder
          .template AddSystem<drake::systems::sensors::ImageToLcmImageArrayT>();
  image_to_lcm_image_array->set_name("converter");

  const auto& cam_port =
      image_to_lcm_image_array
          ->DeclareImageInputPort<drake::systems::sensors::PixelType::kRgba8U>(
              camera_name);
  builder.Connect(camera->color_image_output_port(), cam_port);
  auto image_array_lcm_publisher = builder.template AddSystem
      (drake::systems::lcm::LcmPublisherSystem::Make<robotlocomotion
      ::image_array_t>("DRAKE_RGBD_CAMERA_IMAGES", nullptr, 1.0/10 /*
 * 10 fps publish period*/));
  builder.Connect(image_to_lcm_image_array->image_array_t_msg_output_port(),
                  image_array_lcm_publisher->get_input_port());

  // Build the builder.
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();

  // Create a context for this diagram and plant.
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);
  return 0;
}

}  // namespace
}  // namespace coke_cam
}  // namespace examples
}  // namespace drake

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::coke_cam::do_main();
}
