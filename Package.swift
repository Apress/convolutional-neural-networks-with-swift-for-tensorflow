// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "ConvolutionalNeuralNetworksWithSwiftForTensorFlow",
  platforms: [
    .macOS(.v10_13),
  ],
  dependencies: [
    .package(
      name: "swift-models", url: "https://github.com/tensorflow/swift-models.git", .branch("main")
    ),
  ],
  targets: [
    .target(
      name: "MNIST-1D", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "MNIST-1D"),
    .target(
      name: "MNIST-2D", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "MNIST-2D"),
    .target(
      name: "CIFAR", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "CIFAR"),
    .target(
      name: "VGG", dependencies: [.product(name: "Datasets", package: "swift-models")], path: "VGG"),
    .target(
      name: "Resnet34", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "Resnet34"),
    .target(
      name: "Resnet50", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "Resnet50"),
    .target(
      name: "SqueezeNet", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "SqueezeNet"),
    .target(
      name: "MobileNetV1", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "MobileNetV1"),
    .target(
      name: "MobileNetV2", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "MobileNetV2"),
    .target(
      name: "EfficientNet", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "EfficientNet"),
    .target(
      name: "MobileNetV3", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "MobileNetV3"),
    .target(
      name: "MNIST-XLA-TPU", dependencies: [.product(name: "Datasets", package: "swift-models")],
      path: "MNIST-XLA-TPU"),
  ]
)
