#include <memory>

#include <gflags/gflags.h>

#include "drake/examples/pr2/sim_world.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/primitives/constant_vector_source.h"

namespace drake {
namespace examples {
namespace pr2 {
namespace {

DEFINE_double(simulation_time, 20.0,
              "Desired duration of the simulation in seconds");

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

int DoMain() {
  // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;
  auto sim_world = builder.AddSystem<SimWorld<double>>("pr2");

  geometry::ConnectDrakeVisualizer(&builder,
                                   sim_world->get_mutable_scene_graph(),
                                   sim_world->GetOutputPort("pose_bundle"));

  multibody::ConnectContactResultsToDrakeVisualizer(
      &builder, sim_world->get_mutable_sim_world_plant(),
      sim_world->GetOutputPort("contact_results"));

  // Add a constant source to set desired position and velocity. These values
  // are set arbitrarily.
  const auto& pr2_plant = sim_world->get_robot_plant();
  drake::VectorX<double> constant_pos_value = drake::VectorX<double>::Zero(
      pr2_plant.num_positions() + pr2_plant.num_velocities());

  auto desired_pos_constant_source =
      builder.template AddSystem<systems::ConstantVectorSource<double>>(
          constant_pos_value);
  desired_pos_constant_source->set_name("desired_pos_constant_source");
  builder.Connect(desired_pos_constant_source->get_output_port(),
                  sim_world->GetInputPort("robot_desired_state"));

  auto diagram = builder.Build();

  // Create and run the simulator.
  drake::systems::Simulator<double> simulator(*diagram,
                                              diagram->CreateDefaultContext());
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace pr2
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  return drake::examples::pr2::DoMain();
}
