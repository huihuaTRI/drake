# PR2 Example
This example wants to provide a reference of using different controllers for different parts of a robot. With defining
a robot config parameter structure, it allows the capability to design controllers on a part base. This example implements
two controllers. One simple PID controller for the robot base and one inverse dynamics controller for the upper body.
Using the robot parameter configuration file, this example can be extended to a wide variety of robots that have this
requirement. It also shows an example of how to use Drake yaml utility to load parameters from file.

## Example

### Run visualizer
Before running the example, launch the visualizer:
```
bazel-bin/tools/drake_visualizer
```

### Run simulation
```
bazel-bin/examples/pr2/pr2_simulation
```

This simulates a PR2 holding the zero position using two controllers for different parts of the robot.
In particular, the wheels are controlled by a simple PID controller and the upper body is controlled
using the Drake inverse dynamics controller.