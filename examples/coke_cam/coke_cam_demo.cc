/// Simulate a camera that views a coke can

#include <limits>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/benchmarks/inclined_plane/inclined_plane_plant.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

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
//
static const char* const kCokeCanUrdfPath =
    "drake/examples/coke_cam/urdf/coke_can.urdf";

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

  const std::string full_name = FindResourceOrThrow(kCokeCanUrdfPath);

  auto coke_model_instance_index = parser.AddModelFromFile(full_name);
  (void)coke_model_instance_index;

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
