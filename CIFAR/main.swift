import Datasets
import TensorFlow

struct CIFARModel: Layer {
  var conv1a = Conv2D<Float>(filterShape: (3, 3, 3, 32), padding: .same, activation: relu)
  var conv1b = Conv2D<Float>(filterShape: (3, 3, 32, 32), padding: .same, activation: relu)
  var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  var conv2a = Conv2D<Float>(filterShape: (3, 3, 32, 64), padding: .same, activation: relu)
  var conv2b = Conv2D<Float>(filterShape: (3, 3, 64, 64), padding: .same, activation: relu)
  var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  var flatten = Flatten<Float>()
  var inputLayer = Dense<Float>(inputSize: 8 * 8 * 64, outputSize: 512, activation: relu)
  var hiddenLayer = Dense<Float>(inputSize: 512, outputSize: 512, activation: relu)
  var outputLayer = Dense<Float>(inputSize: 512, outputSize: 10)

  @differentiable
  func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let conv1 = input.sequenced(through: conv1a, conv1b, pool1)
    let conv2 = conv1.sequenced(through: conv2a, conv2b, pool2)
    return conv2.sequenced(through: flatten, inputLayer, hiddenLayer, outputLayer)
  }
}

let batchSize = 128
let epochCount = 12
var model = CIFARModel()
let optimizer = SGD(for: model, learningRate: 0.1)
let dataset = CIFAR10(batchSize: batchSize)

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
    totalGuessCount = totalGuessCount + batch.data.shape[0]
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
