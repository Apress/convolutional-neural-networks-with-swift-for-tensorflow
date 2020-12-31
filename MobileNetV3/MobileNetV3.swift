import TensorFlow

public enum ActivationType {
  case hardSwish
  case relu
}

public struct SqueezeExcitationBlock: Layer {
  // https://arxiv.org/abs/1709.01507
  public var averagePool = GlobalAvgPool2D<Float>()
  public var reduceConv: Conv2D<Float>
  public var expandConv: Conv2D<Float>
  @noDerivative public var inputOutputSize: Int

  public init(inputOutputSize: Int, reducedSize: Int) {
    self.inputOutputSize = inputOutputSize
    reduceConv = Conv2D<Float>(
      filterShape: (1, 1, inputOutputSize, reducedSize),
      strides: (1, 1),
      padding: .same)
    expandConv = Conv2D<Float>(
      filterShape: (1, 1, reducedSize, inputOutputSize),
      strides: (1, 1),
      padding: .same)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let avgPoolReshaped = averagePool(input).reshaped(to: [
      input.shape[0], 1, 1, self.inputOutputSize,
    ])
    return input
      * hardSigmoid(expandConv(relu(reduceConv(avgPoolReshaped))))
  }
}

public struct InitialInvertedResidualBlock: Layer {
  @noDerivative public var addResLayer: Bool
  @noDerivative public var useSELayer: Bool = false
  @noDerivative public var activation: ActivationType = .relu

  public var dConv: DepthwiseConv2D<Float>
  public var batchNormDConv: BatchNorm<Float>
  public var seBlock: SqueezeExcitationBlock
  public var conv2: Conv2D<Float>
  public var batchNormConv2: BatchNorm<Float>

  public init(
    filters: (Int, Int),
    strides: (Int, Int) = (1, 1),
    kernel: (Int, Int) = (3, 3),
    seLayer: Bool = false,
    activation: ActivationType = .relu
  ) {
    self.useSELayer = seLayer
    self.activation = activation
    self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

    let filterMult = filters
    let hiddenDimension = filterMult.0 * 1
    let reducedDimension = hiddenDimension / 4

    dConv = DepthwiseConv2D<Float>(
      filterShape: (3, 3, filterMult.0, 1),
      strides: (1, 1),
      padding: .same)
    seBlock = SqueezeExcitationBlock(
      inputOutputSize: hiddenDimension, reducedSize: reducedDimension)
    conv2 = Conv2D<Float>(
      filterShape: (1, 1, hiddenDimension, filterMult.1),
      strides: (1, 1),
      padding: .same)
    batchNormDConv = BatchNorm(featureCount: filterMult.0)
    batchNormConv2 = BatchNorm(featureCount: filterMult.1)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    var depthwise = batchNormDConv(dConv(input))
    switch self.activation {
    case .hardSwish: depthwise = hardSwish(depthwise)
    case .relu: depthwise = relu(depthwise)
    }

    var squeezeExcite: Tensor<Float>
    if self.useSELayer {
      squeezeExcite = seBlock(depthwise)
    } else {
      squeezeExcite = depthwise
    }

    let piecewiseLinear = batchNormConv2(conv2(squeezeExcite))

    if self.addResLayer {
      return input + piecewiseLinear
    } else {
      return piecewiseLinear
    }
  }
}

public struct InvertedResidualBlock: Layer {
  @noDerivative public var strides: (Int, Int)
  @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  @noDerivative public var addResLayer: Bool
  @noDerivative public var activation: ActivationType = .relu
  @noDerivative public var useSELayer: Bool

  public var conv1: Conv2D<Float>
  public var batchNormConv1: BatchNorm<Float>
  public var dConv: DepthwiseConv2D<Float>
  public var batchNormDConv: BatchNorm<Float>
  public var seBlock: SqueezeExcitationBlock
  public var conv2: Conv2D<Float>
  public var batchNormConv2: BatchNorm<Float>

  public init(
    filters: (Int, Int),
    expansionFactor: Float,
    strides: (Int, Int) = (1, 1),
    kernel: (Int, Int) = (3, 3),
    seLayer: Bool = false,
    activation: ActivationType = .relu
  ) {
    self.strides = strides
    self.addResLayer = filters.0 == filters.1 && strides == (1, 1)
    self.useSELayer = seLayer
    self.activation = activation

    let filterMult = filters
    let hiddenDimension = Int(Float(filterMult.0) * expansionFactor)
    let reducedDimension = hiddenDimension / 4

    conv1 = Conv2D<Float>(
      filterShape: (1, 1, filterMult.0, hiddenDimension),
      strides: (1, 1),
      padding: .same)
    dConv = DepthwiseConv2D<Float>(
      filterShape: (kernel.0, kernel.1, hiddenDimension, 1),
      strides: strides,
      padding: strides == (1, 1) ? .same : .valid)
    seBlock = SqueezeExcitationBlock(
      inputOutputSize: hiddenDimension, reducedSize: reducedDimension)
    conv2 = Conv2D<Float>(
      filterShape: (1, 1, hiddenDimension, filterMult.1),
      strides: (1, 1),
      padding: .same)
    batchNormConv1 = BatchNorm(featureCount: hiddenDimension)
    batchNormDConv = BatchNorm(featureCount: hiddenDimension)
    batchNormConv2 = BatchNorm(featureCount: filterMult.1)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    var piecewise = batchNormConv1(conv1(input))
    switch self.activation {
    case .hardSwish: piecewise = hardSwish(piecewise)
    case .relu: piecewise = relu(piecewise)
    }
    var depthwise: Tensor<Float>
    if self.strides == (1, 1) {
      depthwise = batchNormDConv(dConv(piecewise))
    } else {
      depthwise = batchNormDConv(dConv(zeroPad(piecewise)))
    }
    switch self.activation {
    case .hardSwish: depthwise = hardSwish(depthwise)
    case .relu: depthwise = relu(depthwise)
    }
    var squeezeExcite: Tensor<Float>
    if self.useSELayer {
      squeezeExcite = seBlock(depthwise)
    } else {
      squeezeExcite = depthwise
    }

    let piecewiseLinear = batchNormConv2(conv2(squeezeExcite))

    if self.addResLayer {
      return input + piecewiseLinear
    } else {
      return piecewiseLinear
    }
  }
}

public struct MobileNetV3Large: Layer {
  @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  public var inputConv: Conv2D<Float>
  public var inputConvBatchNorm: BatchNorm<Float>

  public var invertedResidualBlock1: InitialInvertedResidualBlock
  public var invertedResidualBlock2: InvertedResidualBlock
  public var invertedResidualBlock3: InvertedResidualBlock
  public var invertedResidualBlock4: InvertedResidualBlock
  public var invertedResidualBlock5: InvertedResidualBlock
  public var invertedResidualBlock6: InvertedResidualBlock
  public var invertedResidualBlock7: InvertedResidualBlock
  public var invertedResidualBlock8: InvertedResidualBlock
  public var invertedResidualBlock9: InvertedResidualBlock
  public var invertedResidualBlock10: InvertedResidualBlock
  public var invertedResidualBlock11: InvertedResidualBlock
  public var invertedResidualBlock12: InvertedResidualBlock
  public var invertedResidualBlock13: InvertedResidualBlock
  public var invertedResidualBlock14: InvertedResidualBlock
  public var invertedResidualBlock15: InvertedResidualBlock

  public var outputConv: Conv2D<Float>
  public var outputConvBatchNorm: BatchNorm<Float>

  public var avgPool = GlobalAvgPool2D<Float>()
  public var finalConv: Conv2D<Float>
  public var dropoutLayer: Dropout<Float>
  public var classiferConv: Conv2D<Float>
  public var flatten = Flatten<Float>()

  @noDerivative public var lastConvChannel: Int

  public init(classCount: Int = 1000, dropout: Double = 0.2) {
    inputConv = Conv2D<Float>(
      filterShape: (3, 3, 3, 16),
      strides: (2, 2),
      padding: .same)
    inputConvBatchNorm = BatchNorm(
      featureCount: 16)

    invertedResidualBlock1 = InitialInvertedResidualBlock(
      filters: (16, 16))
    invertedResidualBlock2 = InvertedResidualBlock(
      filters: (16, 24),
      expansionFactor: 4, strides: (2, 2))
    invertedResidualBlock3 = InvertedResidualBlock(
      filters: (24, 24),
      expansionFactor: 3)
    invertedResidualBlock4 = InvertedResidualBlock(
      filters: (24, 40),
      expansionFactor: 3, strides: (2, 2), kernel: (5, 5), seLayer: true)
    invertedResidualBlock5 = InvertedResidualBlock(
      filters: (40, 40),
      expansionFactor: 3, kernel: (5, 5), seLayer: true)
    invertedResidualBlock6 = InvertedResidualBlock(
      filters: (40, 40),
      expansionFactor: 3, kernel: (5, 5), seLayer: true)
    invertedResidualBlock7 = InvertedResidualBlock(
      filters: (40, 80),
      expansionFactor: 6, strides: (2, 2), activation: .hardSwish)
    invertedResidualBlock8 = InvertedResidualBlock(
      filters: (80, 80),
      expansionFactor: 2.5, activation: .hardSwish)
    invertedResidualBlock9 = InvertedResidualBlock(
      filters: (80, 80),
      expansionFactor: 184 / 80.0, activation: .hardSwish)
    invertedResidualBlock10 = InvertedResidualBlock(
      filters: (80, 80),
      expansionFactor: 184 / 80.0, activation: .hardSwish)
    invertedResidualBlock11 = InvertedResidualBlock(
      filters: (80, 112),
      expansionFactor: 6, seLayer: true, activation: .hardSwish)
    invertedResidualBlock12 = InvertedResidualBlock(
      filters: (112, 112),
      expansionFactor: 6, seLayer: true, activation: .hardSwish)
    invertedResidualBlock13 = InvertedResidualBlock(
      filters: (112, 160),
      expansionFactor: 6, strides: (2, 2), kernel: (5, 5), seLayer: true,
      activation: .hardSwish)
    invertedResidualBlock14 = InvertedResidualBlock(
      filters: (160, 160),
      expansionFactor: 6, kernel: (5, 5), seLayer: true, activation: .hardSwish)
    invertedResidualBlock15 = InvertedResidualBlock(
      filters: (160, 160),
      expansionFactor: 6, kernel: (5, 5), seLayer: true, activation: .hardSwish)

    lastConvChannel = 960
    outputConv = Conv2D<Float>(
      filterShape: (
        1, 1, 160, lastConvChannel
      ),
      strides: (1, 1),
      padding: .same)
    outputConvBatchNorm = BatchNorm(featureCount: lastConvChannel)

    let lastPointChannel = 1280
    finalConv = Conv2D<Float>(
      filterShape: (1, 1, lastConvChannel, lastPointChannel),
      strides: (1, 1),
      padding: .same)
    dropoutLayer = Dropout<Float>(probability: dropout)
    classiferConv = Conv2D<Float>(
      filterShape: (1, 1, lastPointChannel, classCount),
      strides: (1, 1),
      padding: .same)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let initialConv = hardSwish(
      input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
    let backbone1 = initialConv.sequenced(
      through: invertedResidualBlock1,
      invertedResidualBlock2, invertedResidualBlock3, invertedResidualBlock4,
      invertedResidualBlock5)
    let backbone2 = backbone1.sequenced(
      through: invertedResidualBlock6, invertedResidualBlock7,
      invertedResidualBlock8, invertedResidualBlock9, invertedResidualBlock10)
    let backbone3 = backbone2.sequenced(
      through: invertedResidualBlock11,
      invertedResidualBlock12, invertedResidualBlock13, invertedResidualBlock14,
      invertedResidualBlock15)
    let outputConvResult = hardSwish(outputConvBatchNorm(outputConv(backbone3)))
    let averagePool = avgPool(outputConvResult).reshaped(to: [
      input.shape[0], 1, 1, self.lastConvChannel,
    ])
    let finalConvResult = dropoutLayer(hardSwish(finalConv(averagePool)))
    return flatten(classiferConv(finalConvResult))
  }
}

public struct MobileNetV3Small: Layer {
  @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  public var inputConv: Conv2D<Float>
  public var inputConvBatchNorm: BatchNorm<Float>

  public var invertedResidualBlock1: InitialInvertedResidualBlock
  public var invertedResidualBlock2: InvertedResidualBlock
  public var invertedResidualBlock3: InvertedResidualBlock
  public var invertedResidualBlock4: InvertedResidualBlock
  public var invertedResidualBlock5: InvertedResidualBlock
  public var invertedResidualBlock6: InvertedResidualBlock
  public var invertedResidualBlock7: InvertedResidualBlock
  public var invertedResidualBlock8: InvertedResidualBlock
  public var invertedResidualBlock9: InvertedResidualBlock
  public var invertedResidualBlock10: InvertedResidualBlock
  public var invertedResidualBlock11: InvertedResidualBlock

  public var outputConv: Conv2D<Float>
  public var outputConvBatchNorm: BatchNorm<Float>

  public var avgPool = GlobalAvgPool2D<Float>()
  public var finalConv: Conv2D<Float>
  public var dropoutLayer: Dropout<Float>
  public var classiferConv: Conv2D<Float>
  public var flatten = Flatten<Float>()

  @noDerivative public var lastConvChannel: Int

  public init(classCount: Int = 1000, dropout: Double = 0.2) {
    inputConv = Conv2D<Float>(
      filterShape: (3, 3, 3, 16),
      strides: (2, 2),
      padding: .same)
    inputConvBatchNorm = BatchNorm(
      featureCount: 16)

    invertedResidualBlock1 = InitialInvertedResidualBlock(
      filters: (16, 16),
      strides: (2, 2), seLayer: true)
    invertedResidualBlock2 = InvertedResidualBlock(
      filters: (16, 24),
      expansionFactor: 72.0 / 16.0, strides: (2, 2))
    invertedResidualBlock3 = InvertedResidualBlock(
      filters: (24, 24),
      expansionFactor: 88.0 / 24.0)
    invertedResidualBlock4 = InvertedResidualBlock(
      filters: (24, 40),
      expansionFactor: 4, strides: (2, 2), kernel: (5, 5), seLayer: true,
      activation: .hardSwish)
    invertedResidualBlock5 = InvertedResidualBlock(
      filters: (40, 40),
      expansionFactor: 6, kernel: (5, 5), seLayer: true, activation: .hardSwish)
    invertedResidualBlock6 = InvertedResidualBlock(
      filters: (40, 40),
      expansionFactor: 6, kernel: (5, 5), seLayer: true, activation: .hardSwish)
    invertedResidualBlock7 = InvertedResidualBlock(
      filters: (40, 48),
      expansionFactor: 3, kernel: (5, 5), seLayer: true, activation: .hardSwish)
    invertedResidualBlock8 = InvertedResidualBlock(
      filters: (48, 48),
      expansionFactor: 3, kernel: (5, 5), seLayer: true, activation: .hardSwish)
    invertedResidualBlock9 = InvertedResidualBlock(
      filters: (48, 96),
      expansionFactor: 6, strides: (2, 2), kernel: (5, 5), seLayer: true,
      activation: .hardSwish)
    invertedResidualBlock10 = InvertedResidualBlock(
      filters: (96, 96),
      expansionFactor: 6, kernel: (5, 5), seLayer: true, activation: .hardSwish)
    invertedResidualBlock11 = InvertedResidualBlock(
      filters: (96, 96),
      expansionFactor: 6, kernel: (5, 5), seLayer: true, activation: .hardSwish)

    lastConvChannel = 576
    outputConv = Conv2D<Float>(
      filterShape: (
        1, 1, 96, lastConvChannel
      ),
      strides: (1, 1),
      padding: .same)
    outputConvBatchNorm = BatchNorm(featureCount: lastConvChannel)

    let lastPointChannel = 1280
    finalConv = Conv2D<Float>(
      filterShape: (1, 1, lastConvChannel, lastPointChannel),
      strides: (1, 1),
      padding: .same)
    dropoutLayer = Dropout<Float>(probability: dropout)
    classiferConv = Conv2D<Float>(
      filterShape: (1, 1, lastPointChannel, classCount),
      strides: (1, 1),
      padding: .same)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let initialConv = hardSwish(
      input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
    let backbone1 = initialConv.sequenced(
      through: invertedResidualBlock1,
      invertedResidualBlock2, invertedResidualBlock3, invertedResidualBlock4,
      invertedResidualBlock5)
    let backbone2 = backbone1.sequenced(
      through: invertedResidualBlock6, invertedResidualBlock7,
      invertedResidualBlock8, invertedResidualBlock9, invertedResidualBlock10,
      invertedResidualBlock11)
    let outputConvResult = hardSwish(outputConvBatchNorm(outputConv(backbone2)))
    let averagePool = avgPool(outputConvResult).reshaped(to: [
      input.shape[0], 1, 1, lastConvChannel,
    ])
    let finalConvResult = dropoutLayer(hardSwish(finalConv(averagePool)))
    return flatten(classiferConv(finalConvResult))
  }
}
