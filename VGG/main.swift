import Datasets
import TensorFlow

struct VGG16: Layer {
  var conv1a = Conv2D<Float>(filterShape: (3, 3, 3, 64), padding: .same, activation: relu)
  var conv1b = Conv2D<Float>(filterShape: (3, 3, 64, 64), padding: .same, activation: relu)
  var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  var conv2a = Conv2D<Float>(filterShape: (3, 3, 64, 128), padding: .same, activation: relu)
  var conv2b = Conv2D<Float>(filterShape: (3, 3, 128, 128), padding: .same, activation: relu)
  var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  var conv3a = Conv2D<Float>(filterShape: (3, 3, 128, 256), padding: .same, activation: relu)
  var conv3b = Conv2D<Float>(filterShape: (3, 3, 256, 256), padding: .same, activation: relu)
  var conv3c = Conv2D<Float>(filterShape: (3, 3, 256, 256), padding: .same, activation: relu)
  var pool3 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  var conv4a = Conv2D<Float>(filterShape: (3, 3, 256, 512), padding: .same, activation: relu)
  var conv4b = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
  var conv4c = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
  var pool4 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  var conv5a = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
  var conv5b = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
  var conv5c = Conv2D<Float>(filterShape: (3, 3, 512, 512), padding: .same, activation: relu)
  var pool5 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  var flatten = Flatten<Float>()
  var inputLayer = Dense<Float>(inputSize: 512 * 7 * 7, outputSize: 4096, activation: relu)
  var hiddenLayer = Dense<Float>(inputSize: 4096, outputSize: 4096, activation: relu)
  var outputLayer = Dense<Float>(inputSize: 4096, outputSize: 10)

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let conv1 = input.sequenced(through: conv1a, conv1b, pool1)
    let conv2 = conv1.sequenced(through: conv2a, conv2b, pool2)
    let conv3 = conv2.sequenced(through: conv3a, conv3b, conv3c, pool3)
    let conv4 = conv3.sequenced(through: conv4a, conv4b, conv4c, pool4)
    let conv5 = conv4.sequenced(through: conv5a, conv5b, conv5c, pool5)
    return conv5.sequenced(through: flatten, inputLayer, hiddenLayer, outputLayer)
  }
}

let batchSize = 32
let epochCount = 10

let dataset = Imagenette(batchSize: batchSize, inputSize: .resized320, outputSize: 224)
var model = VGG16()
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
