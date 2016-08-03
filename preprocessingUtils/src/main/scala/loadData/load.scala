package preprocessingUtils.loadData

import scala.reflect.ClassTag
import scala.io.Source

import com.esotericsoftware.kryo.io.Input
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.{mllib, SparkContext}

import preprocessingUtils.DataPoint
import loadLibSVM._
import loadSpaces._

object load {

  /**
   * If the provided data file(s) is/are text files, this function returns training and test set
   * as an RDD containing elements of type Array[Double].
   * 
   * @param sc Spark context
   * @param dataFile Path to data file if only one data file is provided
   * @param nPartitions Minimal number of partitions to use for resulting RDDs.  
   * @param textDataFormat Can be "libsvm", "spaces", or "comma"
   * @param separateTrainTestFiles True if training and test file are provided separately
   * @param trainingDatafile Path to training data file if provided separately from test file
   * @param testDatafile Path to test data file if provided separately from training file
   * @param proportionTest If only one data file is provided, proportion of observations to 
   *                       use for testing.
   * @param seed Random seed
   * @return A tuple containing two RDD[ Array[Double] ] containing the training data resp. the 
   *         test data.
   */
  def readTextFiles(
      sc: SparkContext,
      dataFile: String,
      nPartitions: Int,
      textDataFormat : String,
      sparse : Boolean,
      separateTrainTestFiles : Boolean,
      trainingDatafile : String,
      testDatafile : String,
      proportionTest : Double,
      seed : Int) : (RDD[LabeledPoint], RDD[LabeledPoint]) = {

    // load file(s) and if needed, split into training and test set
    textDataFormat match {
        case "libsvm" =>
          // OPTION 1: LIBSVM format
          if (!separateTrainTestFiles) {
            val splits =
              loadLibSVMFile(sc, dataFile, nPartitions, sparse)
                .randomSplit(Array(1.0 - proportionTest, proportionTest), seed = seed)
            (splits(0), splits(1))
          } else {
            (loadLibSVMFile(sc, trainingDatafile, nPartitions, sparse),
              loadLibSVMFile(sc, testDatafile, nPartitions, sparse))
          }
        case "spaces" =>
          // OPTION 2: First entry on each line is the response,
          // separated by a space from the features, features are separated by spaces
          if (!separateTrainTestFiles) {
            val splits =
              loadSpaceSeparatedFile_overRows(sc, dataFile, nPartitions, sparse)
                .randomSplit(Array(1.0 - proportionTest, proportionTest), seed = seed)
            (splits(0), splits(1))
          } else {
            (loadSpaceSeparatedFile_overRows(sc, trainingDatafile, nPartitions, sparse),
              loadSpaceSeparatedFile_overRows(sc, testDatafile, nPartitions, sparse))
          }
        case "comma" =>
          // OPTION 3: First entry on each line is the response, separated by a comma from the
          // features, features are separated by spaces
          if (!separateTrainTestFiles) {
            val splits =
              loadResponseWithCommaSeparatedFile_overRows(sc, dataFile, nPartitions, sparse)
                .randomSplit(Array(1.0 - proportionTest, proportionTest), seed = seed)
            (splits(0), splits(1))
          } else {
            (loadResponseWithCommaSeparatedFile_overRows(sc, trainingDatafile, nPartitions, sparse),
              loadResponseWithCommaSeparatedFile_overRows(sc, testDatafile, nPartitions, sparse))
          }
        case _ => throw new Error("No such text file option! textDataFormat must be either " +
          "'libsvm', 'spaces' or 'comma'.")
    }
  }

  /*
  Read response vector after saving with '.toArray.mkString(" ")'
   */
  def readResponse(responsePath: String) : mllib.linalg.Vector = {
    // load data
    val responseString = Source.fromFile(responsePath).getLines().flatMap(x => x.split(" ")).map(_.toDouble).toArray
    Vectors.dense(responseString)
  }

  /*
  Read response vector after saving with '.toArray.mkString(" ")'
   */
  def readPartitionsFile(partitionsPath: String)  = {
    // load data
    Source.fromFile(partitionsPath).getLines().flatMap(x => x.split(" ")).map(_.toInt).toArray.zipWithIndex

  }


  /**
   * If the provided data file(s) is/are object files
   * (created with saveAsObjectFile), this function returns training and test set.
   *
   * @param sc Spark context
   * @param dataFile Path to data file if only one data file is provided
   * @param nPartitions Minimal number of partitions to use for resulting RDDs.  
   * @param separateTrainTestFiles True if training and test file are provided separately
   * @param trainingDatafile Path to training data file if provided separately from test file
   * @param testDatafile Path to test data file if provided separately from training file
   * @param proportionTest If only one data file is provided, proportion of observations to 
   *                       use for testing.
   * @param seed Random seed
   * @return A tuple containing two RDD[ Array[Double] ] containing the training data resp. the 
   *         test data.
   */
  def readObjectFiles[T](
      sc: SparkContext,
      dataFile: String,
      nPartitions: Int,
      separateTrainTestFiles : Boolean,
      trainingDatafile : String,
      testDatafile : String,
      proportionTest : Double,
      seed : Int)(implicit ct: ClassTag[T]) : (RDD[T], RDD[T]) = {

    // OPTION 4: Data saved by saveAsObjectFile
    if (!separateTrainTestFiles) {
      val splits =
        objectFile[T](sc, dataFile, minPartitions = nPartitions)
          .randomSplit(Array(1.0 - proportionTest, proportionTest), seed = seed)
      (splits(0), splits(1))
    } else {
      (objectFile[T](sc, trainingDatafile, minPartitions = nPartitions),
        objectFile[T](sc, testDatafile, minPartitions = nPartitions))
    }
  }

  /** Object file using Kryo serialization. */
  def objectFile[T](
      sc: SparkContext,
      path: String,
      minPartitions: Int = 1)(implicit ct: ClassTag[T]) = {

    val kryoSerializer = new KryoSerializer(sc.getConf)

    sc
      .sequenceFile(path, classOf[NullWritable], classOf[BytesWritable], minPartitions)
      .flatMap(x => {
        val kryo = kryoSerializer.newKryo()
        val input = new Input()
        input.setBuffer(x._2.getBytes)
        val data = kryo.readClassAndObject(input)
        val dataObject = data.asInstanceOf[Array[T]]
        dataObject
        }
      )
  }

  /**
   * Maps an RDD containing arrays of doubles where the response is the respective first
   * entry to an RDD containing elements of type LabeledPoint.
   *
   * @param data RDD containing arrays of doubles
   * @return RDD containing elements of type LabeledPoint
   */
  def rddDoubleArrayToLabeledPoint(data : RDD[Array[Double]]) : RDD[LabeledPoint] = {
    data.map(x => LabeledPoint(x.head, Vectors.dense(x.tail)))
  }

  /**
   * Maps an RDD containing arrays of doubles where the response is the respective first
   * entry to an RDD containing elements of type LabeledPoint with SparseVectors.
   *
   * @param data RDD containing arrays of doubles
   * @return RDD containing elements of type LabeledPoint
   */
  def rddDoubleArrayToLabeledPointSparse(data : RDD[Array[Double]]) : RDD[LabeledPoint] = {
    data.map(x => LabeledPoint(x.head, Vectors.dense(x.tail).toSparse))
  }

  /**
   * Maps an RDD containing arrays of doubles where the response is the respective first
   * entry to an RDD containing elements of type DataPoint.
   *
   * @param data RDD containing arrays of doubles
   * @return RDD containing elements of type DataPoint
   */
  def rddDoubleArrayToDataPoint(data : RDD[Array[Double]]) : RDD[DataPoint] = {
    data.map(x => DataPoint(x.head, breeze.linalg.Vector(x.tail)))
  }

  /**
   * Maps an RDD containing elements of type LabeledPoint to an RDD
   * containing elements of type DataPoint.
   *
   * @param data RDD containing elements of type LabeledPoint
   * @return RDD containing elements of type DataPoint
   */
  def rddLabeledPointToDataPoint(data : RDD[LabeledPoint]) : RDD[DataPoint] = {
    data.map(x => DataPoint(x.label, breeze.linalg.Vector(x.features.toArray)))
  }

  /** Maps an array of doubles where the response is the first entry to a LabeledPoint. */
  def doubleArrayToLabeledPoint(point : Array[Double]) : LabeledPoint = {
    LabeledPoint(point.head, Vectors.dense(point.tail))
  }

  /** Maps an array of doubles where the response is the first entry to a LabeledPoint with
    * sparse observations. */
  def doubleArrayToLabeledPointSparse(point : Array[Double]) : LabeledPoint = {
    LabeledPoint(point.head, Vectors.dense(point.tail).toSparse)
  }

  /** Maps an array of doubles where the response is the first entry to a DataPoint. */
  def doubleArrayToDataPoint(point : Array[Double]) : DataPoint = {
   DataPoint(point.head, breeze.linalg.Vector(point.tail))
  }

  /** Map a DataPoint to a LabeledPoint. */
  def dataPointToLabeledPoint(point : DataPoint) : LabeledPoint = {
    LabeledPoint(point.label, Vectors.dense(point.features.toArray))
  }

  /** Map a DataPoint to a LabeledPoint with sparse observations.. */
  def dataPointToLabeledPointSparse(point : DataPoint) : LabeledPoint = {
    LabeledPoint(point.label, Vectors.dense(point.features.toArray).toSparse)
  }

  /** Map a LabeledPoint to a DataPoint. */
  def labeledPointToDataPoint(point : LabeledPoint) : DataPoint = {
    DataPoint(point.label, breeze.linalg.Vector(point.features.toArray))
  }

  /** Map a LabeledPoint to an array of doubles where the response is the first entry. */
  def labeledPointToDoubleArray(point : LabeledPoint) : Array[Double] = {
   Array(point.label) ++ point.features.toArray
  }

  /** Map a LabeledPoint dense to a LabeledPoint sparse. */
  def labeledPointDenseToLabeledPointSparse(point : LabeledPoint) : LabeledPoint = {
    LabeledPoint(point.label, point.features.toSparse)
  }

  /** Map a LabeledPoint sparse to a LabeledPoint dense. */
  def labeledPointSparseToLabeledPointDense(point : LabeledPoint) : LabeledPoint = {
    LabeledPoint(point.label, point.features.toDense)
  }

  /** Cast an Array[Double] or a DataPoint to a LabeledPoint. */
  def castToLabeledPoint[T](x : T): LabeledPoint = {

    x match {
      case a : Array[Double] => doubleArrayToLabeledPoint(x.asInstanceOf[Array[Double]])
      case b : DataPoint => dataPointToLabeledPoint(x.asInstanceOf[DataPoint])
      case c : LabeledPoint => labeledPointSparseToLabeledPointDense(x.asInstanceOf[LabeledPoint])
      case _ => throw new Error("Input type has to be an RDD containing elements of type" +
        "Array[Double] or DataPoint!")
    }

  }

  /** Cast an Array[Double] or a DataPoint to a LabeledPoint sparse. */
  def castToLabeledPointSparse[T](x : T): LabeledPoint = {

    x match {
      case a : Array[Double] => doubleArrayToLabeledPointSparse(x.asInstanceOf[Array[Double]])
      case b : DataPoint => dataPointToLabeledPointSparse(x.asInstanceOf[DataPoint])
      case c : LabeledPoint => labeledPointDenseToLabeledPointSparse(x.asInstanceOf[LabeledPoint])
      case _ => throw new Error("Input type has to be an RDD containing elements of type" +
        "Array[Double] or DataPoint!")
    }

  }

  /** Reads object files(s) and casts the elements to LabeledPoints. */
  def readAndCastObjectFiles[T](
      sc: SparkContext,
      dataFile: String,
      nPartitions: Int,
      sparse : Boolean,
      separateTrainTestFiles : Boolean,
      trainingDatafile : String,
      testDatafile : String,
      proportionTest : Double,
      seed : Int)(implicit ct: ClassTag[T]) : (RDD[LabeledPoint], RDD[LabeledPoint]) = {


    val (train : RDD[T], test : RDD[T]) =
      readObjectFiles[T](
        sc, dataFile, nPartitions, separateTrainTestFiles, trainingDatafile, testDatafile,
        proportionTest, seed)

    if(sparse)
      (train.map(x => castToLabeledPointSparse[T](x)), test.map(x => castToLabeledPointSparse[T](x)))
    else
      (train.map(x => castToLabeledPoint[T](x)), test.map(x => castToLabeledPoint[T](x)))


  }
}
