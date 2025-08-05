Kotlin
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.optimize.api.TrainingListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*

class MachineLearningNotifier(private val model: MultiLayerNetwork) {

    fun train(dataIterator: DataSetIterator) {
        model.setListeners(object : TrainingListener {
            override fun onEpoch(epoch: Int, model: MultiLayerNetwork) {
                println("Epoch $epoch complete")
            }
        })
        model.fit(dataIterator)
    }

    fun notifyModelCompletion() {
        println("Machine learning model training complete!")
        // Add notification logic here (e.g. send email, notification, etc.)
    }
}

fun main() {
    val seed = 123
    val rng = Nd4j.getRandom()
    rng.setSeed(seed)

    val conf = NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                    .nIn(784)
                    .nOut(250)
                    .activation(Activation.RELU)
                    .build())
            .layer(1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(250)
                    .nOut(10)
                    .activation(Activation.SOFTMAX)
                    .build())
            .pretrain(false)
            .backprop(true)
            .build()

    val model = MultiLayerNetwork(conf)
    model.init()

    val notifier = MachineLearningNotifier(model)
    val dataIterator = // create dataset iterator
    notifier.train(dataIterator)
    notifier.notifyModelCompletion()
}