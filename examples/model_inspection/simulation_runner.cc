#include <memory>
#include <utility>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/geometry_instance.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/render/render_engine_vtk_factory.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/shape_specification.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/sensors/image_to_lcm_image_array_t.h"
#include "drake/systems/sensors/pixel_types.h"
#include "drake/systems/sensors/rgbd_sensor.h"

DEFINE_double(simulation_time, 10.0,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 3e-3,
              "The time step to use for MultibodyPlant model "
              "discretization.  0 uses the continuous version of the plant.");
// User could change the following parameters for different models and different
// view angles.
DEFINE_double(camera_position_x, 0.3, "x [m] position of the camera");
DEFINE_double(camera_position_y, -2.0, "y [m] position of the camera");
DEFINE_double(camera_position_z, 0.5, "z [m] position of the camera");
DEFINE_double(camera_rotation_x, -1.5, "x [rad] rotation of the camera");
DEFINE_double(camera_rotation_y, 0.0, "y [rad] rotation of the camera");
DEFINE_double(camera_rotation_z, 0.0, "z [rad] rotation of the camera");
DEFINE_string(model_path,
              "drake/examples/model_inspection/multiple_objects.sdf",
              "File path of the to be loaded model.");
DEFINE_string(model_root_link, "house_root",
              "The root link of the to be loaded model.");

namespace drake {
namespace examples {
namespace model_inspection {
namespace {

using Eigen::Vector3d;
using Eigen::Vector4d;
using geometry::ConnectDrakeVisualizer;
using geometry::GeometryId;
using geometry::GeometryInstance;
using geometry::HalfSpace;
using geometry::IllustrationProperties;
using geometry::PerceptionProperties;
using geometry::ProximityProperties;
using geometry::SceneGraph;
using geometry::SourceId;
using geometry::render::DepthCameraProperties;
using geometry::render::RenderEngineVtkParams;
using geometry::render::RenderLabel;
using lcm::DrakeLcm;
using math::RigidTransformd;
using math::RollPitchYawd;
using math::RotationMatrixd;
using multibody::ConnectContactResultsToDrakeVisualizer;
using multibody::MultibodyPlant;
using multibody::Parser;
using std::make_unique;
using systems::InputPort;
using systems::sensors::PixelType;
using systems::sensors::RgbdSensor;

const Eigen::Vector3d kCameraPosition{0.3, -2, 1.0};

int do_main() {
  systems::DiagramBuilder<double> builder;
  auto scene_graph = builder.AddSystem<SceneGraph<double>>();
  scene_graph->set_name("scene_graph");
  const std::string render_name("renderer");
  scene_graph->AddRenderer(render_name,
                           MakeRenderEngineVtk(RenderEngineVtkParams()));

  // Load the model from sdf/urdf.
  MultibodyPlant<double>* model_plant =
      builder.AddSystem<MultibodyPlant>(FLAGS_time_step);
  model_plant->RegisterAsSourceForSceneGraph(scene_graph);

  Parser(model_plant, scene_graph)
      .AddModelFromFile(FindResourceOrThrow(FLAGS_model_path));
  model_plant->WeldFrames(model_plant->world_frame(),
                          model_plant->GetFrameByName(FLAGS_model_root_link));
  model_plant->Finalize();

  builder.Connect(
      model_plant->get_geometry_poses_output_port(),
      scene_graph->get_source_pose_port(model_plant->get_source_id().value()));
  builder.Connect(scene_graph->get_query_output_port(),
                  model_plant->get_geometry_query_input_port());

  // Add a camera to test the rendering.
  // Create the camera.
  DepthCameraProperties camera_properties(640, 480, M_PI_2, render_name, 0.1,
                                          10.0);
  // Set camera position and the direction it looks at.
  const Vector3d p_WB(FLAGS_camera_position_x, FLAGS_camera_position_y,
                      FLAGS_camera_position_z);
  const RollPitchYawd R_WB(FLAGS_camera_rotation_x, FLAGS_camera_rotation_y,
                           FLAGS_camera_rotation_z);
  const RigidTransformd X_WB(R_WB, p_WB);

  auto camera = builder.AddSystem<RgbdSensor>(scene_graph->world_frame_id(),
                                              X_WB, camera_properties);
  builder.Connect(scene_graph->get_query_output_port(),
                  camera->query_object_input_port());

  // Broadcast the images.
  // Publishing images to drake visualizer
  auto image_to_lcm_image_array =
      builder.template AddSystem<systems::sensors::ImageToLcmImageArrayT>();
  image_to_lcm_image_array->set_name("converter");

  systems::lcm::LcmPublisherSystem* image_array_lcm_publisher{nullptr};

  DrakeLcm lcm;
  ConnectDrakeVisualizer(&builder, *scene_graph, &lcm);
  ConnectContactResultsToDrakeVisualizer(
      &builder, *model_plant, model_plant->get_contact_results_output_port());

  image_array_lcm_publisher = builder.template AddSystem(
      systems::lcm::LcmPublisherSystem::Make<robotlocomotion::image_array_t>(
          "DRAKE_RGBD_CAMERA_IMAGES", &lcm, 0.1 /* publish period */));
  image_array_lcm_publisher->set_name("publisher");

  builder.Connect(image_to_lcm_image_array->image_array_t_msg_output_port(),
                  image_array_lcm_publisher->get_input_port());

  {
    const auto& port =
        image_to_lcm_image_array->DeclareImageInputPort<PixelType::kRgba8U>(
            "color");
    builder.Connect(camera->color_image_output_port(), port);
  }

  {
    const auto& port =
        image_to_lcm_image_array->DeclareImageInputPort<PixelType::kDepth32F>(
            "depth");
    builder.Connect(camera->depth_image_32F_output_port(), port);
  }

  {
    const auto& port =
        image_to_lcm_image_array->DeclareImageInputPort<PixelType::kLabel16I>(
            "label");
    builder.Connect(camera->label_image_output_port(), port);
  }

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1.f);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace model_inspection
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::model_inspection::do_main();
}
