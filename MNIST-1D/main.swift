// 1
import Datasets
import TensorFlow

// 2
struct MLP: Layer {
  var flatten = Flatten<Float>()
  var inputLayer = Dense<Float>(inputSize: 784, outputSize: 512, activation: relu)
  var hiddenLayer = Dense<Float>(inputSize: 512, outputSize: 512, activation: relu)
  var outputLayer = Dense<Float>(inputSize: 512, outputSize: 10)

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    return input.sequenced(through: flatten, inputLayer, hiddenLayer, outputLayer)
  }
}

// 3
let batchSize = 128
let epochCount = 12
var model = MLP()
let optimizer = SGD(for: model, learningRate: 0.1)
let dataset = MNIST(batchSize: batchSize)

print("Starting training...")

for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
  // 4
  Context.local.learningPhase = .training
  for batch in epochBatches {
    let (images, labels) = (batch.data, batch.label)
    let (_, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
      let logits = model(images)
      return softmaxCrossEntropy(logits: logits, labels: labels)
    }
    optimizer.update(&model, along: gradients)
  }

  // 5
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
