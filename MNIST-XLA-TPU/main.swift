import Datasets
import TensorFlow

struct CNN: Layer {
  var conv1a = Conv2D<Float>(filterShape: (3, 3, 1, 32), padding: .same, activation: relu)
  var conv1b = Conv2D<Float>(filterShape: (3, 3, 32, 32), padding: .same, activation: relu)
  var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  var flatten = Flatten<Float>()
  var inputLayer = Dense<Float>(inputSize: 14 * 14 * 32, outputSize: 512, activation: relu)
  var hiddenLayer = Dense<Float>(inputSize: 512, outputSize: 512, activation: relu)
  var outputLayer = Dense<Float>(inputSize: 512, outputSize: 10)

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let convolutionLayer = input.sequenced(through: conv1a, conv1b, pool1)
    return convolutionLayer.sequenced(through: flatten, inputLayer, hiddenLayer, outputLayer)
  }
}

let batchSize = 128
let epochCount = 12
var model = CNN()
var optimizer = SGD(for: model, learningRate: 0.1)
let dataset = MNIST(batchSize: batchSize)

let device = Device.defaultXLA
model.move(to: device)
optimizer = SGD(copying: optimizer, to: device)

print("Starting training...")

for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
  Context.local.learningPhase = .training
  for batch in epochBatches {
    let (images, labels) = (batch.data, batch.label)
    let deviceImages = Tensor(copying: images, to: device)
    let deviceLabels = Tensor(copying: labels, to: device)
    let (_, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
      let logits = model(deviceImages)
      return softmaxCrossEntropy(logits: logits, labels: deviceLabels)
    }
    optimizer.update(&model, along: gradients)
    LazyTensorBarrier()
  }

  Context.local.learningPhase = .inference
  var testLossSum: Float = 0
  var testBatchCount = 0
  var correctGuessCount = 0
  var totalGuessCount = 0

  for batch in dataset.validation {
    let (images, labels) = (batch.data, batch.label)
    let deviceImages = Tensor(copying: images, to: device)
    let deviceLabels = Tensor(copying: labels, to: device)
    let logits = model(deviceImages)
    testLossSum += softmaxCrossEntropy(logits: logits, labels: deviceLabels).scalarized()
    testBatchCount += 1

    let correctPredictions = logits.argmax(squeezingAxis: 1) .== deviceLabels
    correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
    totalGuessCount = totalGuessCount + batch.data.shape[0]
    LazyTensorBarrier()
  }

  let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
  print(
    """
    [Epoch \(epoch + 1)] \
    Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
    Loss: \(testLossSum / Float(testBatchCount))
    """
  )
}
