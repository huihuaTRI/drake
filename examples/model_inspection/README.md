
## This folder provides tools and examples to inspect a model
### Each subfolder in this directory contains an example of a model that will work with Drake.
For example:
- **exterior_door_2020-02-14_01** shows an example of having multiple links in only one urdf/sdf file
- **tea_bottle** shows an example of only one link

> To point out some high-level requirements about generating models to work with Drake
> - Each link (or body) must have proper mass and inertia. Sometimes it may be hard to get these information accurately.
> Rough estimation would still be good. At least, identity mass and inertia matrix should be provided.
> - Texture image has to be prepared properly. For the texture png to show in the Drake visualizer, the visualization
> `.obj` has to reference an `.mtl` file, which then references the texture image file.
> - Proper contact geometry is also very important for dynamics related simulation. Drake supports only geometry primitives
> (box, sphere, capsule, etc,) and convex `.objs`.
> - For models with articulated joints, the axes and motion range of each joint have to be properly defined.

### Model inspection
To inspect the geometry properties and the joint axes. Use the following command:
> - First, build the tool via running `bazel build //manipulation/util:geometry_inspector`
> - Run `./bazel-bin/manipulation/util/geometry_inspector ./examples/model_inspection/tea_bottle/tea_bottle.sdf` This command only
> shows the visual geometry (including texture images).
> - `./bazel-bin/manipulation/util/geometry_inspector --visualize_collisions ./examples/model_inspection/tea_bottle/tea_bottle.sdf`
> This command will show both the visual and the contact geometries. The contact geometries will be displayed in red with `50%` transparency.

Note that, your models have to be included in the build system properly. Refer to `//examples/model_inspection/BUILD.bazel` for examples.

To inspect the dynamics and perception properties of each model, use the the following command:
> - First, build the simulation via `bazel build //examples/model_inspection:simulation_runner`,
> - Run `./bazel-bin/examples/model_inspection/simulation_runner --options`. Please refer to the `simulation_runner.cc` for the
> details about options. To change the camera x and y positions, you could `./bazel-bin/examples/model_inspection/simulation_runner --camera_position_x=0.1 --camera_position_y=0.4`

This routine will run a dynamics simulation with the model you specified. It will check the mass, inertial etc. It also has a camera
to make sure the texture images are loaded properly for perception purpose, i.e., the camera image should also show the textures.

**To summarize, each model should pass these two inspections before we can conclude.**

### Fuse multiple models into one single sdf
The file **multiple_objects.sdf** shows an example of combining multiple models into one urdf/sdf file. The general steps are
> - Create a `model_root` link which serves as the root link of the whole urdf/sdf.
> - Copy and paste the links and joints definition of each individual model to this single urdf/sdf file.
> - Add a `fixed` joint between the `model_root` link and the `root` link of each individual model