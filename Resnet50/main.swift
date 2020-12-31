import Datasets
import TensorFlow

struct ConvBN: Layer {
  var conv: Conv2D<Float>
  var norm: BatchNorm<Float>

  init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid
  ) {
    self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
    self.norm = BatchNorm(featureCount: filterShape.3)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    return input.sequenced(through: conv, norm)
  }
}

struct ResidualConvBlock: Layer {
  var layer1: ConvBN
  var layer2: ConvBN
  var layer3: ConvBN
  var shortcut: ConvBN

  init(
    featureCounts: (Int, Int, Int, Int),
    kernelSize: Int = 3,
    strides: (Int, Int) = (2, 2)
  ) {
    self.layer1 = ConvBN(
      filterShape: (1, 1, featureCounts.0, featureCounts.1),
      strides: strides)
    self.layer2 = ConvBN(
      filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
      padding: .same)
    self.layer3 = ConvBN(filterShape: (1, 1, featureCounts.2, featureCounts.3))
    self.shortcut = ConvBN(
      filterShape: (1, 1, featureCounts.0, featureCounts.3),
      strides: strides,
      padding: .same)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let tmp = relu(layer2(relu(layer1(input))))
    return relu(layer3(tmp) + shortcut(input))
  }
}

struct ResidualIdentityBlock: Layer {
  var layer1: ConvBN
  var layer2: ConvBN
  var layer3: ConvBN

  init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
    self.layer1 = ConvBN(filterShape: (1, 1, featureCounts.0, featureCounts.1))
    self.layer2 = ConvBN(
      filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
      padding: .same)
    self.layer3 = ConvBN(filterShape: (1, 1, featureCounts.2, featureCounts.3))
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let tmp = relu(layer2(relu(layer1(input))))
    return relu(layer3(tmp) + input)
  }
}

struct ResNet50: Layer {
  var l1: ConvBN
  var maxPool: MaxPool2D<Float>

  var l2a = ResidualConvBlock(featureCounts: (64, 64, 64, 256), strides: (1, 1))
  var l2b = ResidualIdentityBlock(featureCounts: (256, 64, 64, 256))
  var l2c = ResidualIdentityBlock(featureCounts: (256, 64, 64, 256))

  var l3a = ResidualConvBlock(featureCounts: (256, 128, 128, 512))
  var l3b = ResidualIdentityBlock(featureCounts: (512, 128, 128, 512))
  var l3c = ResidualIdentityBlock(featureCounts: (512, 128, 128, 512))
  var l3d = ResidualIdentityBlock(featureCounts: (512, 128, 128, 512))

  var l4a = ResidualConvBlock(featureCounts: (512, 256, 256, 1024))
  var l4b = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))
  var l4c = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))
  var l4d = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))
  var l4e = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))
  var l4f = ResidualIdentityBlock(featureCounts: (1024, 256, 256, 1024))

  var l5a = ResidualConvBlock(featureCounts: (1024, 512, 512, 2048))
  var l5b = ResidualIdentityBlock(featureCounts: (2048, 512, 512, 2048))
  var l5c = ResidualIdentityBlock(featureCounts: (2048, 512, 512, 2048))

  var avgPool: AvgPool2D<Float>
  var flatten = Flatten<Float>()
  var classifier: Dense<Float>

  init() {
    l1 = ConvBN(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
    maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
    avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
    classifier = Dense(inputSize: 2048, outputSize: 10)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let inputLayer = maxPool(relu(l1(input)))
    let level2 = inputLayer.sequenced(through: l2a, l2b, l2c)
    let level3 = level2.sequenced(through: l3a, l3b, l3c, l3d)
    let level4 = level3.sequenced(through: l4a, l4b, l4c, l4d, l4e, l4f)
    let level5 = level4.sequenced(through: l5a, l5b, l5c)
    return level5.sequenced(through: avgPool, flatten, classifier)
  }
}

let batchSize = 32
let epochCount = 30

let dataset = Imagenette(batchSize: batchSize, inputSize: .resized320, outputSize: 224)
var model = ResNet50()
let optimizer = SGD(for: model, learningRate: 0.002, momentum: 0.9)

print("Starting training...")

for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
  Context.local.learningPhase = .training
  for batch in epochBatches {
    let (images, labels) = (batch.data, batch.label)
    let (_, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
      let logits = model(images)
      return softmaxCrossEntropy(logits: logits, labels: labels)
    }
    optimizer.update(&model, along: gradients)
  }

  Context.local.learningPhase = .inference
  var testLossSum: Float = 0
  var testBatchCount = 0
  var correctGuessCount = 0
  var totalGuessCount = 0
  for batch in dataset.validation {
    let (images, labels) = (batch.data, batch.label)
    let logits = model(images)
    testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
    testBatchCount += 1

    let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
    correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
    totalGuessCount = totalGuessCount + batch.label.shape[0]
  }

  let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
  print(
    """
    [Epoch \(epoch+1)] \
    Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
    Loss: \(testLossSum / Float(testBatchCount))
    """
  )
}
