import TensorFlow

public struct Fire: Layer {
  public var squeeze: Conv2D<Float>
  public var expand1: Conv2D<Float>
  public var expand3: Conv2D<Float>

  public init(
    inputFilterCount: Int,
    squeezeFilterCount: Int,
    expand1FilterCount: Int,
    expand3FilterCount: Int
  ) {
    squeeze = Conv2D(
      filterShape: (1, 1, inputFilterCount, squeezeFilterCount),
      activation: relu)
    expand1 = Conv2D(
      filterShape: (1, 1, squeezeFilterCount, expand1FilterCount),
      activation: relu)
    expand3 = Conv2D(
      filterShape: (3, 3, squeezeFilterCount, expand3FilterCount),
      padding: .same,
      activation: relu)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let squeezed = squeeze(input)
    let expanded1 = expand1(squeezed)
    let expanded3 = expand3(squeezed)
    return expanded1.concatenated(with: expanded3, alongAxis: -1)
  }
}

public struct SqueezeNet: Layer {
  public var inputConv = Conv2D<Float>(
    filterShape: (3, 3, 3, 64),
    strides: (2, 2),
    padding: .same,
    activation: relu)
  public var maxPool1 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
  public var fire2 = Fire(
    inputFilterCount: 64,
    squeezeFilterCount: 16,
    expand1FilterCount: 64,
    expand3FilterCount: 64)
  public var fire3 = Fire(
    inputFilterCount: 128,
    squeezeFilterCount: 16,
    expand1FilterCount: 64,
    expand3FilterCount: 64)
  public var maxPool3 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
  public var fire4 = Fire(
    inputFilterCount: 128,
    squeezeFilterCount: 32,
    expand1FilterCount: 128,
    expand3FilterCount: 128)
  public var fire5 = Fire(
    inputFilterCount: 256,
    squeezeFilterCount: 32,
    expand1FilterCount: 128,
    expand3FilterCount: 128)
  public var maxPool5 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2))
  public var fire6 = Fire(
    inputFilterCount: 256,
    squeezeFilterCount: 48,
    expand1FilterCount: 192,
    expand3FilterCount: 192)
  public var fire7 = Fire(
    inputFilterCount: 384,
    squeezeFilterCount: 48,
    expand1FilterCount: 192,
    expand3FilterCount: 192)
  public var fire8 = Fire(
    inputFilterCount: 384,
    squeezeFilterCount: 64,
    expand1FilterCount: 256,
    expand3FilterCount: 256)
  public var fire9 = Fire(
    inputFilterCount: 512,
    squeezeFilterCount: 64,
    expand1FilterCount: 256,
    expand3FilterCount: 256)
  public var outputConv: Conv2D<Float>
  public var avgPool = AvgPool2D<Float>(poolSize: (13, 13), strides: (1, 1))

  public init(classCount: Int = 10) {
    outputConv = Conv2D(filterShape: (1, 1, 512, classCount), strides: (1, 1), activation: relu)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let convolved1 = input.sequenced(through: inputConv, maxPool1)
    let fired1 = convolved1.sequenced(through: fire2, fire3, maxPool3, fire4, fire5)
    let fired2 = fired1.sequenced(through: maxPool5, fire6, fire7, fire8, fire9)
    let output = fired2.sequenced(through: outputConv, avgPool)
    return output.reshaped(to: [input.shape[0], outputConv.filter.shape[3]])
  }
}
