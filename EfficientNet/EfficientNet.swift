import TensorFlow

struct InitialMBConvBlock: Layer {
  @noDerivative var hiddenDimension: Int
  var dConv: DepthwiseConv2D<Float>
  var batchNormDConv: BatchNorm<Float>
  var seAveragePool = GlobalAvgPool2D<Float>()
  var seReduceConv: Conv2D<Float>
  var seExpandConv: Conv2D<Float>
  var conv2: Conv2D<Float>
  var batchNormConv2: BatchNorm<Float>

  init(filters: (Int, Int), width: Float) {
    let filterMult = filters
    self.hiddenDimension = filterMult.0
    dConv = DepthwiseConv2D<Float>(
      filterShape: (3, 3, filterMult.0, 1),
      strides: (1, 1),
      padding: .same)
    seReduceConv = Conv2D<Float>(
      filterShape: (1, 1, filterMult.0, 8),
      strides: (1, 1),
      padding: .same)
    seExpandConv = Conv2D<Float>(
      filterShape: (1, 1, 8, filterMult.0),
      strides: (1, 1),
      padding: .same)
    conv2 = Conv2D<Float>(
      filterShape: (1, 1, filterMult.0, filterMult.1),
      strides: (1, 1),
      padding: .same)
    batchNormDConv = BatchNorm(featureCount: filterMult.0)
    batchNormConv2 = BatchNorm(featureCount: filterMult.1)
  }

  @differentiable
  func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let depthwise = swish(batchNormDConv(dConv(input)))
    let seAvgPoolReshaped = seAveragePool(depthwise).reshaped(to: [
      input.shape[0], 1, 1, self.hiddenDimension,
    ])
    let squeezeExcite =
      depthwise
      * sigmoid(seExpandConv(swish(seReduceConv(seAvgPoolReshaped))))
    return batchNormConv2(conv2(squeezeExcite))
  }
}

struct MBConvBlock: Layer {
  @noDerivative var addResLayer: Bool
  @noDerivative var strides: (Int, Int)
  @noDerivative let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  @noDerivative var hiddenDimension: Int

  var conv1: Conv2D<Float>
  var batchNormConv1: BatchNorm<Float>
  var dConv: DepthwiseConv2D<Float>
  var batchNormDConv: BatchNorm<Float>
  var seAveragePool = GlobalAvgPool2D<Float>()
  var seReduceConv: Conv2D<Float>
  var seExpandConv: Conv2D<Float>
  var conv2: Conv2D<Float>
  var batchNormConv2: BatchNorm<Float>

  init(
    filters: (Int, Int),
    width: Float,
    depthMultiplier: Int = 6,
    strides: (Int, Int) = (1, 1),
    kernel: (Int, Int) = (3, 3)
  ) {
    self.strides = strides
    self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

    let filterMult = filters
    self.hiddenDimension = filterMult.0 * depthMultiplier
    let reducedDimension = max(1, Int(filterMult.0 / 4))
    conv1 = Conv2D<Float>(
      filterShape: (1, 1, filterMult.0, hiddenDimension),
      strides: (1, 1),
      padding: .same)
    dConv = DepthwiseConv2D<Float>(
      filterShape: (kernel.0, kernel.1, hiddenDimension, 1),
      strides: strides,
      padding: strides == (1, 1) ? .same : .valid)
    seReduceConv = Conv2D<Float>(
      filterShape: (1, 1, hiddenDimension, reducedDimension),
      strides: (1, 1),
      padding: .same)
    seExpandConv = Conv2D<Float>(
      filterShape: (1, 1, reducedDimension, hiddenDimension),
      strides: (1, 1),
      padding: .same)
    conv2 = Conv2D<Float>(
      filterShape: (1, 1, hiddenDimension, filterMult.1),
      strides: (1, 1),
      padding: .same)
    batchNormConv1 = BatchNorm(featureCount: hiddenDimension)
    batchNormDConv = BatchNorm(featureCount: hiddenDimension)
    batchNormConv2 = BatchNorm(featureCount: filterMult.1)
  }

  @differentiable
  func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let piecewise = swish(batchNormConv1(conv1(input)))
    var depthwise: Tensor<Float>
    if self.strides == (1, 1) {
      depthwise = swish(batchNormDConv(dConv(piecewise)))
    } else {
      depthwise = swish(batchNormDConv(dConv(zeroPad(piecewise))))
    }
    let seAvgPoolReshaped = seAveragePool(depthwise).reshaped(to: [
      input.shape[0], 1, 1, self.hiddenDimension,
    ])
    let squeezeExcite =
      depthwise
      * sigmoid(seExpandConv(swish(seReduceConv(seAvgPoolReshaped))))
    let piecewiseLinear = batchNormConv2(conv2(squeezeExcite))

    if self.addResLayer {
      return input + piecewiseLinear
    } else {
      return piecewiseLinear
    }
  }
}

struct MBConvBlockStack: Layer {
  var blocks: [MBConvBlock] = []

  init(
    filters: (Int, Int),
    width: Float,
    initialStrides: (Int, Int) = (2, 2),
    kernel: (Int, Int) = (3, 3),
    blockCount: Int,
    depth: Float
  ) {
    let blockMult = blockCount
    self.blocks = [
      MBConvBlock(
        filters: (filters.0, filters.1), width: width,
        strides: initialStrides, kernel: kernel)
    ]
    for _ in 1..<blockMult {
      self.blocks.append(
        MBConvBlock(
          filters: (filters.1, filters.1),
          width: width, kernel: kernel))
    }
  }

  @differentiable
  func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    return blocks.differentiableReduce(input) { $1($0) }
  }
}

public struct EfficientNet: Layer {
  @noDerivative let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
  var inputConv: Conv2D<Float>
  var inputConvBatchNorm: BatchNorm<Float>
  var initialMBConv: InitialMBConvBlock

  var residualBlockStack1: MBConvBlockStack
  var residualBlockStack2: MBConvBlockStack
  var residualBlockStack3: MBConvBlockStack
  var residualBlockStack4: MBConvBlockStack
  var residualBlockStack5: MBConvBlockStack
  var residualBlockStack6: MBConvBlockStack

  var outputConv: Conv2D<Float>
  var outputConvBatchNorm: BatchNorm<Float>
  var avgPool = GlobalAvgPool2D<Float>()
  var dropoutProb: Dropout<Float>
  var outputClassifier: Dense<Float>

  public init(
    classCount: Int = 1000,
    width: Float = 1.0,
    depth: Float = 1.0,
    resolution: Int = 224,
    dropout: Double = 0.2
  ) {
    inputConv = Conv2D<Float>(
      filterShape: (3, 3, 3, 32),
      strides: (2, 2),
      padding: .valid)
    inputConvBatchNorm = BatchNorm(featureCount: 32)

    initialMBConv = InitialMBConvBlock(filters: (32, 16), width: width)

    residualBlockStack1 = MBConvBlockStack(
      filters: (16, 24), width: width,
      blockCount: 2, depth: depth)
    residualBlockStack2 = MBConvBlockStack(
      filters: (24, 40), width: width,
      kernel: (5, 5), blockCount: 2, depth: depth)
    residualBlockStack3 = MBConvBlockStack(
      filters: (40, 80), width: width,
      blockCount: 3, depth: depth)
    residualBlockStack4 = MBConvBlockStack(
      filters: (80, 112), width: width,
      initialStrides: (1, 1), kernel: (5, 5), blockCount: 3, depth: depth)
    residualBlockStack5 = MBConvBlockStack(
      filters: (112, 192), width: width,
      kernel: (5, 5), blockCount: 4, depth: depth)
    residualBlockStack6 = MBConvBlockStack(
      filters: (192, 320), width: width,
      initialStrides: (1, 1), blockCount: 1, depth: depth)

    outputConv = Conv2D<Float>(
      filterShape: (
        1, 1,
        320, 1280
      ),
      strides: (1, 1),
      padding: .same)
    outputConvBatchNorm = BatchNorm(featureCount: 1280)

    dropoutProb = Dropout<Float>(probability: dropout)
    outputClassifier = Dense(inputSize: 1280, outputSize: classCount)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let convolved = swish(input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
    let initialBlock = initialMBConv(convolved)
    let backbone = initialBlock.sequenced(
      through: residualBlockStack1, residualBlockStack2,
      residualBlockStack3, residualBlockStack4, residualBlockStack5, residualBlockStack6)
    let output = swish(backbone.sequenced(through: outputConv, outputConvBatchNorm))
    return output.sequenced(through: avgPool, dropoutProb, outputClassifier)
  }
}
