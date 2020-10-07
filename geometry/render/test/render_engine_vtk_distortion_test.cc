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
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkOpenGLTexture.h>
#include <vtkPNGReader.h>
#include <vtkPNGWriter.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkTIFFReader.h>
#include <vtkTIFFWriter.h>

#include "drake/common/drake_copyable.h"
#include "drake/common/filesystem.h"
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
using systems::sensors::Image;
using systems::sensors::ImageDepth32F;
using systems::sensors::ImageLabel16I;
using systems::sensors::ImageRgba8U;
using systems::sensors::InvalidDepth;
using systems::sensors::PixelType;

// Default camera properties.
const int kWidth = 2560;
const int kHeight = 2048;
const double kZNear = 0.5;
const double kZFar = 5.;
const double kFovY = 1.62144727;
const bool kShowWindow = false;

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

template <PixelType kPixelType>
void SaveToFileHelper(const Image<kPixelType>& image,
                      const std::string& file_path) {
  const int width = image.width();
  const int height = image.height();
  const int num_channels = Image<kPixelType>::kNumChannels;

  vtkSmartPointer<vtkImageWriter> writer;
  vtkNew<vtkImageData> vtk_image;
  vtk_image->SetDimensions(width, height, 1);

  // NOTE: This excludes *many* of the defined `PixelType` values.
  switch (kPixelType) {
    case PixelType::kRgba8U:
    case PixelType::kGrey8U:
      vtk_image->AllocateScalars(VTK_UNSIGNED_CHAR, num_channels);
      writer = vtkSmartPointer<vtkPNGWriter>::New();
      break;
    case PixelType::kDepth16U:
      vtk_image->AllocateScalars(VTK_UNSIGNED_SHORT, num_channels);
      writer = vtkSmartPointer<vtkPNGWriter>::New();
      break;
    case PixelType::kDepth32F:
      vtk_image->AllocateScalars(VTK_FLOAT, num_channels);
      writer = vtkSmartPointer<vtkTIFFWriter>::New();
      break;
    case PixelType::kLabel16I:
      vtk_image->AllocateScalars(VTK_UNSIGNED_SHORT, num_channels);
      writer = vtkSmartPointer<vtkPNGWriter>::New();
      break;
    default:
      throw std::logic_error(
          "Unsupported image type; cannot be written to file");
  }

  auto image_ptr = reinterpret_cast<typename Image<kPixelType>::T*>(
      vtk_image->GetScalarPointer());
  const int num_scalar_components = vtk_image->GetNumberOfScalarComponents();
  DRAKE_DEMAND(num_scalar_components == num_channels);

  for (int v = height - 1; v >= 0; --v) {
    for (int u = 0; u < width; ++u) {
      for (int c = 0; c < num_channels; ++c) {
        image_ptr[c] =
            static_cast<typename Image<kPixelType>::T>(image.at(u, v)[c]);
      }
      image_ptr += num_scalar_components;
    }
  }

  writer->SetFileName(file_path.c_str());
  writer->SetInputData(vtk_image.GetPointer());
  writer->Write();
}

template <PixelType kPixelType>
void ReadImage(const std::string& image_name, Image<kPixelType>* image) {
  filesystem::path image_path(image_name);
  if (filesystem::exists(image_path)) {
    vtkSmartPointer<vtkImageReader2> reader;
    switch (kPixelType) {
      case PixelType::kRgba8U:
      case PixelType::kGrey8U:
      case PixelType::kDepth16U:
      case PixelType::kLabel16I:
        reader = vtkSmartPointer<vtkPNGReader>::New();
        break;
      case PixelType::kDepth32F:
        reader = vtkSmartPointer<vtkTIFFReader>::New();
        break;
      default:
        throw std::logic_error("Trying to read an unknown image type");
    }
    reader->SetFileName(image_name.c_str());
    vtkNew<vtkImageExport> exporter;
    exporter->SetInputConnection(reader->GetOutputPort());
    exporter->Update();
    vtkImageData* image_data = exporter->GetInput();
    // Assumes 1-dimensional data -- the 4x1 image.
    std::cout << "Data dimension: " << image_data->GetDataDimension()
              << std::endl;
    if (image_data->GetDataDimension() == 2) {
      int read_width;
      image_data->GetDimensions(&read_width);
      if (read_width == image->width()) {
        exporter->Export(image->at(0, 0));
        return;
      }
    }
    int dims[3];
    exporter->GetDataDimensions(&dims[0]);
    std::cout << "Image dimensions: " << dims[0] << ", " << dims[1] << ", "
              << dims[2] << std::endl;
    throw std::logic_error("The image size does not match.");
  } else {
    throw std::logic_error("The image to be read does not exist.");
  }
}

class DoubleSphereCameraModel {
 public:
  DoubleSphereCameraModel(double fx, double fy, double cx, double cy, double xi,
                          double alpha)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy), xi_(xi), alpha_(alpha) {}

  inline void Point3dToPixel(const Eigen::Vector3d& point,
                             Eigen::Vector2d* pixel) const {
    // Not checking the output pointer due to efficiency.
    double xx = point.x() * point.x();
    double yy = point.y() * point.y();
    double zz = point.z() * point.z();
    double d1 = std::sqrt(xx + yy + zz);
    double z2 = xi_ * d1 + point.z();
    double d2 = std::sqrt(xx + yy + z2 * z2);
    double denominator = alpha_ * d2 + (1 - alpha_) * z2;
    pixel->x() = fx_ * point.x() / denominator + cx_;
    pixel->y() = fy_ * point.y() / denominator + cy_;
  }

  inline void PixelToRay3d(const Eigen::Vector2d& pixel,
                           Eigen::Vector3d* ray) const {
    // Not checking the output pointer due to efficiency.
    double mx = (pixel.x() - cx_) / fx_;
    double my = (pixel.y() - cy_) / fy_;
    double mxx = mx * mx;
    double myy = my * my;
    double rr = mxx + myy;
    double s1 = std::sqrt(1 - (2 * alpha_ - 1) * rr);
    double denominator = alpha_ * s1 + 1 - alpha_;
    double mz = (1 - alpha_ * alpha_ * rr) / denominator;
    double mzz = mz * mz;
    double s2 = std::sqrt(mzz + (1 - xi_ * xi_) * rr);
    double c = (mz * xi_ + s2) / (mzz + rr);
    ray->x() = c * mx;
    ray->y() = c * my;
    ray->z() = c * mz - xi_;
  }

  void ScaleModel(double factor) {
    fx_ *= factor;
    fy_ *= factor;
    cx_ *= factor;
    cy_ *= factor;
  }

  double fx() const { return fx_; }
  double fy() const { return fy_; }
  double cx() const { return cx_; }
  double cy() const { return cy_; }
  double xi() const { return xi_; }
  double alpha() const { return alpha_; }

 private:
  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  double xi_ = 0.0;
  double alpha_ = 0.0;
};

class LinearCameraModel {
 public:
  LinearCameraModel(double fx, double fy, double cx, double cy)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

  inline void Point3dToPixel(const Eigen::Vector3d& point,
                             Eigen::Vector2d* pixel) const {
    pixel->x() = point.x() * fx_ / point.z() + cx_;
    pixel->y() = point.y() * fy_ / point.z() + cy_;
  }

  inline void PixelToRay3d(const Eigen::Vector2d& pixel,
                           Eigen::Vector3d* ray) const {
    ray->x() = (pixel.x() - cx_) / fx_;
    ray->y() = (pixel.y() - cy_) / fy_;
    ray->z() = 1;
  }

  void ScaleModel(double factor) {
    fx_ *= factor;
    fy_ *= factor;
    cx_ *= factor;
    cy_ *= factor;
  }

  double fx() const { return fx_; }
  double fy() const { return fy_; }
  double cx() const { return cx_; }
  double cy() const { return cy_; }

 private:
  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
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

  auto filename = FindResourceOrThrow(
      "drake/geometry/render/test/distortion_test_checkerboard.obj");
  Mesh mesh(filename, 0.2);
  expected_label_ = RenderLabel(4);
  PerceptionProperties material = simple_material();
  material.AddProperty(
      "phong", "diffuse_map",
      FindResourceOrThrow(
          "drake/geometry/render/test/distortion_test_checkerboard.png"));
  const GeometryId id = GeometryId::get_new_id();
  renderer_->RegisterVisual(id, mesh, material, RigidTransformd::Identity(),
                            true /* needs update */);
  renderer_->UpdatePoses(unordered_map<GeometryId, RigidTransformd>{
      {id, RigidTransformd::Identity()}});

  ImageRgba8U color(camera_.width, camera_.height);
  ImageDepth32F depth(camera_.width, camera_.height);
  ImageLabel16I label(camera_.width, camera_.height);
  Render(renderer_.get(), &camera_, &color, &depth, &label);

  const std::string file_path = "/tmp/camera_distortion/color_image.png";
  SaveToFileHelper(color, file_path);

  ImageRgba8U color_read(camera_.width, camera_.height);
  ReadImage(file_path, &color_read);
  // std::promise<void>().get_future().wait_for(std::chrono::seconds(10));
}

TEST_F(RenderEngineVtkTest, ImageDistortionTest) {
  ImageRgba8U color_nodistortion(camera_.width, camera_.height);
  ImageRgba8U color_distorted(camera_.width, camera_.height);

  const std::string no_distortion_image_path =
      "/home/huihuazhao/Pictures/color_image_no_distortion.png";
  const std::string distorted_image_path =
      "/home/huihuazhao/Pictures/color_image_distortion.png";

  ReadImage(no_distortion_image_path, &color_nodistortion);
  ReadImage(distorted_image_path, &color_distorted);

  DoubleSphereCameraModel dd_camera_model(1346.53239902215, 1346.53239902215,
                                          1280, 1024, -0.0715060319640005,
                                          0.702225667086413);

  LinearCameraModel linear_camera_model(1346.53239902215, 1346.53239902215,
                                        1280, 1024);
  for (int i = 0; i < camera_.width; ++i) {
    for (int j = 0; j < camera_.height; ++j) {
    }
  }
}

}  // namespace
}  // namespace render
}  // namespace geometry
}  // namespace drake
