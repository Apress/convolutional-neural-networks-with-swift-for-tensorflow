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

struct ResidualBasicBlock: Layer {
  var layer1: ConvBN
  var layer2: ConvBN

  init(
    featureCounts: (Int, Int, Int, Int),
    kernelSize: Int = 3,
    strides: (Int, Int) = (1, 1)
  ) {
    self.layer1 = ConvBN(
      filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
      strides: strides,
      padding: .same)
    self.layer2 = ConvBN(
      filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.3),
      strides: strides,
      padding: .same)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    return layer2(relu(layer1(input)))
  }
}

struct ResidualBasicBlockShortcut: Layer {
  var layer1: ConvBN
  var layer2: ConvBN
  var shortcut: ConvBN

  init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
    self.layer1 = ConvBN(
      filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
      strides: (2, 2),
      padding: .same)
    self.layer2 = ConvBN(
      filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
      strides: (1, 1),
      padding: .same)
    self.shortcut = ConvBN(
      filterShape: (1, 1, featureCounts.0, featureCounts.3),
      strides: (2, 2),
      padding: .same)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    return layer2(relu(layer1(input))) + shortcut(input)
  }
}

struct ResNet34: Layer {
  var l1: ConvBN
  var maxPool: MaxPool2D<Float>

  var l2a = ResidualBasicBlock(featureCounts: (64, 64, 64, 64))
  var l2b = ResidualBasicBlock(featureCounts: (64, 64, 64, 64))
  var l2c = ResidualBasicBlock(featureCounts: (64, 64, 64, 64))

  var l3a = ResidualBasicBlockShortcut(featureCounts: (64, 128, 128, 128))
  var l3b = ResidualBasicBlock(featureCounts: (128, 128, 128, 128))
  var l3c = ResidualBasicBlock(featureCounts: (128, 128, 128, 128))
  var l3d = ResidualBasicBlock(featureCounts: (128, 128, 128, 128))

  var l4a = ResidualBasicBlockShortcut(featureCounts: (128, 256, 256, 256))
  var l4b = ResidualBasicBlock(featureCounts: (256, 256, 256, 256))
  var l4c = ResidualBasicBlock(featureCounts: (256, 256, 256, 256))
  var l4d = ResidualBasicBlock(featureCounts: (256, 256, 256, 256))
  var l4e = ResidualBasicBlock(featureCounts: (256, 256, 256, 256))
  var l4f = ResidualBasicBlock(featureCounts: (256, 256, 256, 256))

  var l5a = ResidualBasicBlockShortcut(featureCounts: (256, 512, 512, 512))
  var l5b = ResidualBasicBlock(featureCounts: (512, 512, 512, 512))
  var l5c = ResidualBasicBlock(featureCounts: (512, 512, 512, 512))

  var avgPool: AvgPool2D<Float>
  var flatten = Flatten<Float>()
  var classifier: Dense<Float>

  init() {
    l1 = ConvBN(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
    maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
    avgPool = AvgPool2D(poolSize: (7, 7), strides: (7, 7))
    classifier = Dense(inputSize: 512, outputSize: 10)
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
var model = ResNet34()
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
