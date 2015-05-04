package preprocessingUtils.saveData

import java.io.ByteArrayOutputStream

import org.apache.spark.mllib.regression.LabeledPoint
import preprocessingUtils.loadData.load._

import scala.reflect.ClassTag

import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer


object save {

  /** Save RDD as Object file using Kryo serialization. */
  def saveAsObjectFile[T: ClassTag](rdd: RDD[T], path: String) {

    val kryoSerializer = new KryoSerializer(rdd.context.getConf)

    rdd.mapPartitions(iter => iter.grouped(10)
      .map(_.toArray))
      .map(splitArray => {

      //initializes kyro and calls your registrator class
      val kryo = kryoSerializer.newKryo()

      //convert data to bytes
      val bao = new ByteArrayOutputStream()
      val output = kryoSerializer.newKryoOutput()

      output.setOutputStream(bao)
      kryo.writeClassAndObject(output, splitArray)
      output.close()

      // We are ignoring key field of sequence file
      val byteWritable = new BytesWritable(bao.toByteArray)
      (NullWritable.get(), byteWritable)
      }).saveAsSequenceFile(path)
  }


  /**
   * Cast RDD[LabeledPoint] to RDD containing elements of type "outputClass" and save as
   * object file using Kryo serialisation.
   *
   * @param training Training data as RDD[LabeledPoint]
   * @param test Test data as RDD[LabeledPoint]
   * @param outputClass Desired output class, can be "DoubleArray", "DataPoint" or "LabeledPoint"
   * @param outputTrainFileName File name to use for training file, outputClass will be appended to
   *                            this
   * @param outputTestFileName File name to use for test file, outputClass will be appended to this
   */
   def castAndSave(
      training : RDD[LabeledPoint],
      test : RDD[LabeledPoint],
      outputClass : String,
      outputTrainFileName : String,
      outputTestFileName : String) : Unit = {

    // cast in desired format as specified in "outputClass"
    val (trainingData, testData) = outputClass match {
      case "DataPoint" =>
        (training.map(x => labeledPointToDataPoint(x)), test.map(x => labeledPointToDataPoint(x)))
      case "LabeledPoint" => (training, test)
      case "DoubleArray" =>
        (training.map(x => labeledPointToDoubleArray(x)),
          test.map(x => labeledPointToDoubleArray(x)))
      case _ => throw new Error("No such conversion option (must be one of DataPoint, " +
        "LabeledPoint, DoubleArray)!")
    }

    // save data
    save.saveAsObjectFile(trainingData, outputTrainFileName + outputClass)
    save.saveAsObjectFile(testData, outputTestFileName + outputClass)
  }

}
