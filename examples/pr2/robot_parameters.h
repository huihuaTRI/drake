#pragma once

#include <map>
#include <string>
#include <vector>

#include "drake/common/name_value.h"
#include "drake/geometry/render/camera_properties.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace examples {
namespace pr2 {

/// Contains the necessary information about a loaded model instance.
/// This information will be useful if extra care is needed for this instance.
/// For example, the X_PC will be used to set the initial position if this
/// instance has a floating base.
struct ModelInstanceInfo {
  std::string model_name;
  std::string model_path;
  std::string parent_frame_name;
  std::string child_frame_name;
  bool is_floating_base;
  multibody::ModelInstanceIndex index;
  math::RigidTransform<double> X_PC;
};

/// A simple struct to store PID gains for one joint. Serialize the members so
/// that the struct can be loaded from yaml files using the YamlReadArchive.
/// Details can be referred to
/// "examples/pr2/parameters/robot_parameters_loader.h".
struct PidGains {
  double kp{0.0};
  double ki{0.0};
  double kd{0.0};

  bool IsValid() const { return (kp >= 0.0 && ki >= 0.0 && kd >= 0.0); }

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(kp));
    a->Visit(DRAKE_NVP(ki));
    a->Visit(DRAKE_NVP(kd));
  }
};

/// A struct stores the necessary parameters to control a single joint. One
/// joint may have multiple actuators. Serialize the members so that the struct
/// can be loaded from yaml files using the YamlReadArchive.
struct JointParameters {
  std::string name;
  std::string actuator_name;
  double effort_limit = 0.0;  ///< [Nm/rad] for revolute joint, [N/m] for
                              ///< linear joint. Absolute value of the
                              ///< torque limit.
  PidGains gains;

  bool IsValid() const {
    if (!name.empty() && gains.IsValid() && (effort_limit >= 0)) {
      return true;
    }
    return false;
  }

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(name));
    a->Visit(DRAKE_NVP(actuator_name));
    a->Visit(DRAKE_NVP(effort_limit));
    a->Visit(DRAKE_NVP(gains));
  }
};

/// Parameters of the parts that assemble a robot. Parts are defined as the
/// major components of a robot. To be more specific, a part could mean the
/// torso, arm, head, base/chassis, or grippers. Each part may have a different
/// controller strategy. We are only interested in the actuated joints inside
/// a part.
struct PartParameters {
  std::string name;
  std::vector<JointParameters> joints_parameters;

  bool IsValid() const {
    if (name.empty() || joints_parameters.empty()) {
      return false;
    }
    // This map will be used to confirm there are no duplicate joint names
    // (w.r.t joint name) on the same part.
    std::map<std::string, bool> unique_joint_names_per_part_map;
    for (const auto& joint_parameter : joints_parameters) {
      if (!joint_parameter.IsValid()) {
        return false;
      }
      // Return error if duplicate actuator names are found per joint.
      if (unique_joint_names_per_part_map.find(joint_parameter.name) ==
          unique_joint_names_per_part_map.end()) {
        unique_joint_names_per_part_map.insert({joint_parameter.name, true});
      } else {
        return false;
      }
    }
    return true;
  }

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(name));
    a->Visit(DRAKE_NVP(joints_parameters));
  }
};

/// Parameters of a robot that includes different parts and the sensors.
struct RobotParameters {
  std::string name;
  pr2::ModelInstanceInfo model_instance_info;
  std::map<std::string, PartParameters> parts_parameters;

  bool IsValid() const {
    if (name.empty()) {
      return false;
    }
    // These two maps will be used to confirm there are no duplicate part names
    // and joint names of the same robot.
    std::map<std::string, bool> unique_part_names_per_robot_map;
    std::map<std::string, bool> unique_joint_names_per_robot_map;
    for (const auto& [part_name, part_parameter] : parts_parameters) {
      if (part_name.empty() || !part_parameter.IsValid()) {
        return false;
      }
      // Return error if duplicate actuator names are found per joint.
      if (unique_part_names_per_robot_map.find(part_name) ==
          unique_part_names_per_robot_map.end()) {
        unique_part_names_per_robot_map.insert({part_name, true});
      } else {
        return false;
      }

      for (const auto& joint_parameter : part_parameter.joints_parameters) {
        if (!joint_parameter.IsValid()) {
          return false;
        }
        // Return error if duplicate actuator names are found per joint.
        if (unique_joint_names_per_robot_map.find(joint_parameter.name) ==
            unique_joint_names_per_robot_map.end()) {
          unique_joint_names_per_robot_map.insert({joint_parameter.name, true});
        } else {
          return false;
        }
      }
    }
    return true;
  }
};

}  // namespace pr2
}  // namespace examples
}  // namespace drake
