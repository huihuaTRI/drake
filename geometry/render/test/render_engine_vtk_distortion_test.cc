#include "drake/geometry/render/render_engine_vtk.h"

#include <cstring>
#include <future>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <vtkOpenGLTexture.h>
#include <vtkProperty.h>

#include "drake/common/drake_copyable.h"
#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_no_throw.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/geometry/render/camera_properties.h"
#include "drake/geometry/shape_specification.h"
#include "drake/geometry/test_utilities/dummy_render_engine.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/systems/sensors/image.h"

namespace drake {
namespace geometry {
namespace render {
namespace {

using Eigen::AngleAxisd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using geometry::internal::DummyRenderEngine;
using math::RigidTransformd;
using math::RotationMatrixd;
using std::make_unique;
using std::unique_ptr;
using std::unordered_map;
using std::vector;
using systems::sensors::CameraInfo;
using systems::sensors::Color;
using systems::sensors::ColorD;
using systems::sensors::ColorI;
using systems::sensors::ImageDepth32F;
using systems::sensors::ImageLabel16I;
using systems::sensors::ImageRgba8U;
using systems::sensors::InvalidDepth;

// Default camera properties.
const int kWidth = 2560;
const int kHeight = 2048;
const double kZNear = 0.5;
const double kZFar = 5.;
const double kFovY = 1.62144727;
const bool kShowWindow = true;

// The following tolerance is used due to a precision difference between Ubuntu
// Linux and Mac OSX.
const double kColorPixelTolerance = 1.001;
// NOTE: The depth tolerance is this large mostly due to the combination of
// several factors:
//   - the sphere test (sphere against terrain)
//   - The even-valued window dimensions
//   - the tests against various camera properties
// The explanation is as follows. The distance to the sphere is only exactly
// 2 at the point of the sphere directly underneath the camera (the sphere's
// "peak"). However, with an even-valued window dimension, we never really
// sample that point. We sample the center of pixels all evenly arrayed around
// that point. So, that introduces some error. This error is further increased
// in ellipsoid tests when sampling around the elongated ends. As the image gets
// *smaller* the pixels get bigger and so the distance away from the peak center
// increases, which, in turn, increase the measured distance for the fragment.
// This tolerance accounts for the test case where one image has pixels that are
// *4X* larger (in area) than the default image size.
const double kDepthTolerance = 1e-3;

// Background (sky) and terrain colors.
const ColorI kBgColor = {254u, 127u, 0u};
const ColorD kTerrainColorD{0., 0., 0.};
const ColorI kTerrainColorI{0, 0, 0};
// box.png contains a single pixel with the color (4, 241, 33). If the image
// changes, the expected color would likewise have to change.
const ColorI kTextureColor{4, 241, 33};

// Provide a default visual color for these tests -- it is intended to be
// different from the default color of the VTK render engine.
const ColorI kDefaultVisualColor = {229u, 229u, 229u};
const float kDefaultDistance{5.f};

const RenderLabel kDefaultLabel{13531};

// Values to be used with the "centered shape" tests.
// The amount inset from the edge of the images to *still* expect ground plane
// values.
static constexpr int kInset{10};

// Holds `(x, y)` indices of the screen coordinate system where the ranges of
// `x` and `y` are [0, image_width) and [0, image_height) respectively.
struct ScreenCoord {
  int x{};
  int y{};
};

// Utility struct for doing color testing; provides three mechanisms for
// creating a common rgba color. We get colors from images (as a pointer to
// unsigned bytes, as a (ColorI, alpha) pair, and from a normalized color. It's
// nice to articulate tests without having to worry about those details.
struct RgbaColor {
  RgbaColor(const Color<int>& c, int alpha)
      : r(c.r), g(c.g), b(c.b), a(alpha) {}
  explicit RgbaColor(const uint8_t* p) : r(p[0]), g(p[1]), b(p[2]), a(p[3]) {}
  explicit RgbaColor(const Vector4d& norm_color)
      : r(static_cast<int>(norm_color(0) * 255)),
        g(static_cast<int>(norm_color(1) * 255)),
        b(static_cast<int>(norm_color(2) * 255)),
        a(static_cast<int>(norm_color(3) * 255)) {}
  int r;
  int g;
  int b;
  int a;
};

// This test suite facilitates a test with a ground plane and floating shape.
// The camera is positioned above the shape looking straight down. All
// of the images produced from these tests should have the following properties:
//   1. The shape is centered.
//   2. The ground plane fills the whole background (i.e., no background color
//      should be visible), except for noted exceptions.
//   3. The rendered shape should be smaller than the full image size with a
//      minimum number of pixels of ground plane between the shape and the edge
//      of the image. The minimum number of pixels is defined by kInset.
//
// The tests examine the rendered images and tests some discrete pixels, mapped
// to the image size (w, h):
//   1. A "center" pixel (x, y) such that x = w / 2 and y = h / 2.
//   2. Border pixels (xᵢ, yᵢ) which are pixels inset from each corner:
//      e.g., (i, i), (w - i - 1, i), (w - i - 1, h - i - 1), (i, h - i - 1),
//      for an inset value of `i` pixels.
class RenderEngineVtkTest : public ::testing::Test {
 public:
  RenderEngineVtkTest()
      : color_(kWidth, kHeight),
        depth_(kWidth, kHeight),
        label_(kWidth, kHeight),
        // Looking straight down from kDefaultDistance meters above the ground.
        X_WC_(RotationMatrixd{AngleAxisd(M_PI, Vector3d::UnitY()) *
                              AngleAxisd(-M_PI_2, Vector3d::UnitZ())},
              {0, 0.0, kDefaultDistance}),
        geometry_id_(GeometryId::get_new_id()) {}

 protected:
  // Method to allow the normal case (render with the built-in renderer against
  // the default camera) to the member images with default window visibility.
  // This interface allows that to be completely reconfigured by the calling
  // test.
  void Render(RenderEngineVtk* renderer = nullptr,
              const DepthCameraProperties* camera_in = nullptr,
              ImageRgba8U* color_out = nullptr,
              ImageDepth32F* depth_out = nullptr,
              ImageLabel16I* label_out = nullptr) {
    if (!renderer) renderer = renderer_.get();
    const DepthCameraProperties& camera = camera_in ? *camera_in : camera_;
    ImageRgba8U* color = color_out ? color_out : &color_;
    ImageDepth32F* depth = depth_out ? depth_out : &depth_;
    ImageLabel16I* label = label_out ? label_out : &label_;
    renderer->RenderColorImage(camera, kShowWindow, color);
    renderer->RenderDepthImage(camera, depth);
    renderer->RenderLabelImage(camera, false, label);
  }

  // Tests that don't instantiate their own renderers should invoke this.
  void Init(const RigidTransformd& X_WR, bool add_terrain = false) {
    const Vector3d bg_rgb{
        kBgColor.r / 255., kBgColor.g / 255., kBgColor.b / 255.};
    RenderEngineVtkParams params{{}, {}, bg_rgb};
    renderer_ = make_unique<RenderEngineVtk>(params);
    InitializeRenderer(X_WR, add_terrain, renderer_.get());
  }

  // Tests that instantiate their own renderers can initialize their renderers
  // with this method.
  void InitializeRenderer(const RigidTransformd& X_WR, bool add_terrain,
                          RenderEngineVtk* engine) {
    engine->UpdateViewpoint(X_WR);

    if (add_terrain) {
      PerceptionProperties material;
      material.AddProperty("label", "id", RenderLabel::kDontCare);
      material.AddProperty(
          "phong", "diffuse",
          Vector4d{kTerrainColorD.r, kTerrainColorD.g, kTerrainColorD.b, 1.0});
      engine->RegisterVisual(GeometryId::get_new_id(), HalfSpace(), material,
                             RigidTransformd::Identity(),
                             false /* needs update */);
    }
  }

  // Creates a simple perception properties set for fixed, known results. The
  // material color can be modified by setting default_color_ prior to invoking
  // this method.
  PerceptionProperties simple_material(bool use_texture = false) const {
    PerceptionProperties material;
    Vector4d color_n(default_color_.r / 255., default_color_.g / 255.,
                     default_color_.b / 255., default_color_.a / 255.);
    material.AddProperty("phong", "diffuse", color_n);
    material.AddProperty("label", "id", expected_label_);
    if (use_texture) {
      material.AddProperty(
          "phong", "diffuse_map",
          FindResourceOrThrow(
              "drake/systems/sensors/test/models/meshes/box.png"));
    }
    return material;
  }


  // Performs the work to test the rendering with a shape centered in the
  // image. To pass, the renderer will have to have been populated with a
  // compatible shape and camera configuration (e.g., PopulateSphereTest()).
  void PerformCenterShapeTest(RenderEngineVtk* renderer,
                              const char* name,
                              const DepthCameraProperties* camera = nullptr) {
    const DepthCameraProperties& cam = camera ? *camera : camera_;
    // Can't use the member images in case the camera has been configured to a
    // different size than the default camera_ configuration.
    ImageRgba8U color(cam.width, cam.height);
    ImageDepth32F depth(cam.width, cam.height);
    ImageLabel16I label(cam.width, cam.height);
    Render(renderer, &cam, &color, &depth, &label);
  }

  RgbaColor expected_color_{kDefaultVisualColor, 255};
  RgbaColor expected_outlier_color_{kDefaultVisualColor, 255};
  float expected_outlier_depth_{3.f};
  float expected_object_depth_{2.f};
  RenderLabel expected_label_;
  RenderLabel expected_outlier_label_{RenderLabel::kDontCare};
  RgbaColor default_color_{kDefaultVisualColor, 255};

  const DepthCameraProperties camera_ = {kWidth, kHeight, kFovY, "unused",
                                         kZNear, kZFar};

  ImageRgba8U color_;
  ImageDepth32F depth_;
  ImageLabel16I label_;
  RigidTransformd X_WC_;
  GeometryId geometry_id_;

  // The pose of the sphere created in PopulateSphereTest().
  unordered_map<GeometryId, RigidTransformd> X_WV_;

  unique_ptr<RenderEngineVtk> renderer_;
};

// Performs the shape-centered-in-the-image test with a *textured* mesh (which
// happens to be a box).
TEST_F(RenderEngineVtkTest, TextureMeshTest) {
  Init(X_WC_, true);

  auto filename =
      FindResourceOrThrow("drake/geometry/render/test/distortion_test_checkerboard.obj");
  Mesh mesh(filename, 0.2);
  expected_label_ = RenderLabel(4);
  PerceptionProperties material = simple_material();
  material.AddProperty(
      "phong", "diffuse_map",
      FindResourceOrThrow("drake/geometry/render/test/distortion_test_checkerboard.png"));
  const GeometryId id = GeometryId::get_new_id();
  renderer_->RegisterVisual(id, mesh, material, RigidTransformd::Identity(),
                            true /* needs update */);
  renderer_->UpdatePoses(unordered_map<GeometryId, RigidTransformd>{
      {id, RigidTransformd::Identity()}});

  expected_color_ = RgbaColor(kTextureColor, 255);
  PerformCenterShapeTest(renderer_.get(), "Textured mesh test");

  std::promise<void>().get_future().wait();
}

}  // namespace
}  // namespace render
}  // namespace geometry
}  // namespace drake
