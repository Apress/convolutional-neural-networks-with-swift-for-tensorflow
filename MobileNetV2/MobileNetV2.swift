import TensorFlow

public struct InitialInvertedBottleneckBlock: Layer {
  public var dConv: DepthwiseConv2D<Float>
  public var batchNormDConv: BatchNorm<Float>
  public var conv2: Conv2D<Float>
  public var batchNormConv: BatchNorm<Float>

  public init(filters: (Int, Int)) {
    dConv = DepthwiseConv2D<Float>(
      filterShape: (3, 3, filters.0, 1),
      strides: (1, 1),
      padding: .same)
    conv2 = Conv2D<Float>(
      filterShape: (1, 1, filters.0, filters.1),
      strides: (1, 1),
      padding: .same)
    batchNormDConv = BatchNorm(featureCount: filters.0)
    batchNormConv = BatchNorm(featureCount: filters.1)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let depthwise = relu6(batchNormDConv(dConv(input)))
    return batchNormConv(conv2(depthwise))
  }
}

public struct InvertedBottleneckBlock: Layer {
  @noDerivative public var addResLayer: Bool
  @noDerivative public var strides: (Int, Int)
  @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))

  public var conv1: Conv2D<Float>
  public var batchNormConv1: BatchNorm<Float>
  public var dConv: DepthwiseConv2D<Float>
  public var batchNormDConv: BatchNorm<Float>
  public var conv2: Conv2D<Float>
  public var batchNormConv2: BatchNorm<Float>

  public init(
    filters: (Int, Int),
    depthMultiplier: Int = 6,
    strides: (Int, Int) = (1, 1)
  ) {
    self.strides = strides
    self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

    let hiddenDimension = filters.0 * depthMultiplier
    conv1 = Conv2D<Float>(
      filterShape: (1, 1, filters.0, hiddenDimension),
      strides: (1, 1),
      padding: .same)
    dConv = DepthwiseConv2D<Float>(
      filterShape: (3, 3, hiddenDimension, 1),
      strides: strides,
      padding: strides == (1, 1) ? .same : .valid)
    conv2 = Conv2D<Float>(
      filterShape: (1, 1, hiddenDimension, filters.1),
      strides: (1, 1),
      padding: .same)
    batchNormConv1 = BatchNorm(featureCount: hiddenDimension)
    batchNormDConv = BatchNorm(featureCount: hiddenDimension)
    batchNormConv2 = BatchNorm(featureCount: filters.1)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let pointwise = relu6(batchNormConv1(conv1(input)))
    var depthwise: Tensor<Float>
    if self.strides == (1, 1) {
      depthwise = relu6(batchNormDConv(dConv(pointwise)))
    } else {
      depthwise = relu6(batchNormDConv(dConv(zeroPad(pointwise))))
    }
    let pointwiseLinear = batchNormConv2(conv2(depthwise))

    if self.addResLayer {
      return input + pointwiseLinear
    } else {
      return pointwiseLinear
    }
  }
}

public struct InvertedBottleneckBlockStack: Layer {
  var blocks: [InvertedBottleneckBlock] = []

  public init(
    filters: (Int, Int),
    blockCount: Int,
    initialStrides: (Int, Int) = (2, 2)
  ) {
    self.blocks = [
      InvertedBottleneckBlock(
        filters: (filters.0, filters.1),
        strides: initialStrides)
    ]
    for _ in 1..<blockCount {
      self.blocks.append(
        InvertedBottleneckBlock(
          filters: (filters.1, filters.1))
      )
    }
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    return blocks.differentiableReduce(input) { $1($0) }
  }
}

public struct MobileNetV2: Layer {
  @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  public var inputConv: Conv2D<Float>
  public var inputConvBatchNorm: BatchNorm<Float>
  public var initialInvertedBottleneck: InitialInvertedBottleneckBlock

  public var residualBlockStack1: InvertedBottleneckBlockStack
  public var residualBlockStack2: InvertedBottleneckBlockStack
  public var residualBlockStack3: InvertedBottleneckBlockStack
  public var residualBlockStack4: InvertedBottleneckBlockStack
  public var residualBlockStack5: InvertedBottleneckBlockStack

  public var invertedBottleneckBlock16: InvertedBottleneckBlock

  public var outputConv: Conv2D<Float>
  public var outputConvBatchNorm: BatchNorm<Float>
  public var avgPool = GlobalAvgPool2D<Float>()
  public var outputClassifier: Dense<Float>

  public init(classCount: Int = 10) {
    inputConv = Conv2D<Float>(
      filterShape: (3, 3, 3, 32),
      strides: (2, 2),
      padding: .valid)
    inputConvBatchNorm = BatchNorm(
      featureCount: 32)

    initialInvertedBottleneck = InitialInvertedBottleneckBlock(
      filters: (32, 16))

    residualBlockStack1 = InvertedBottleneckBlockStack(filters: (16, 24), blockCount: 2)
    residualBlockStack2 = InvertedBottleneckBlockStack(filters: (24, 32), blockCount: 3)
    residualBlockStack3 = InvertedBottleneckBlockStack(filters: (32, 64), blockCount: 4)
    residualBlockStack4 = InvertedBottleneckBlockStack(
      filters: (64, 96), blockCount: 3,
      initialStrides: (1, 1))
    residualBlockStack5 = InvertedBottleneckBlockStack(filters: (96, 160), blockCount: 3)

    invertedBottleneckBlock16 = InvertedBottleneckBlock(filters: (160, 320))

    outputConv = Conv2D<Float>(
      filterShape: (1, 1, 320, 1280),
      strides: (1, 1),
      padding: .same)
    outputConvBatchNorm = BatchNorm(featureCount: 1280)

    outputClassifier = Dense(inputSize: 1280, outputSize: classCount)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let convolved = relu6(input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
    let initialConv = initialInvertedBottleneck(convolved)
    let backbone = initialConv.sequenced(
      through: residualBlockStack1, residualBlockStack2, residualBlockStack3,
      residualBlockStack4, residualBlockStack5)
    let output = relu6(outputConvBatchNorm(outputConv(invertedBottleneckBlock16(backbone))))
    return output.sequenced(through: avgPool, outputClassifier)
  }
}
