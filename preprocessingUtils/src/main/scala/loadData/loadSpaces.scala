package preprocessingUtils.loadData

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors

object loadSpaces {

  /**
   * Loads a text file with the following structure: First entry on each line is the response,
   * separated by a space from the features, features are separated by spaces
   *
   * @param sc Spark Context
   * @param path Path to text file
   * @param minPartitions minPartition RDD should be partitioned in
   * @return RDD containing elements of type Array[Double] where first element
   *         is the response, followed by the features
   */
  def loadSpaceSeparatedFile_overRows(
      sc: SparkContext,
      path: String,
      minPartitions: Int,
      sparse : Boolean): RDD[LabeledPoint] = {

    // load data
    val data = sc.textFile(path, minPartitions)

    // map each element to Array[Double]
    data.map { line =>
      val observation = line.split(' ').filterNot(elm => elm == "").map(_.toDouble)
      if(sparse)
        LabeledPoint(observation.head, Vectors.dense(observation.tail).toSparse)
      else
        LabeledPoint(observation.head, Vectors.dense(observation.tail))
    }
  }

  /**
   * Loads a text file with the following structure: First entry on each line is the response,
   * separated by a comma from the features, features are separated by spaces
   *
   * @param sc Spark Context
   * @param path Path to text file
   * @param minPartitions minPartition RDD should be partitioned in
   * @return RDD containing elements of type Array[Double] where first element
   *         is the response, followed by the features
   */
  def loadResponseWithCommaSeparatedFile_overRows(
      sc: SparkContext,
      path: String,
      minPartitions: Int,
      sparse : Boolean): RDD[LabeledPoint] = {

    // load data
    val data = sc.textFile(path, minPartitions)

    // map each element to Array[Double]
    data.map { line =>
      val parts = line.split(',')
      val observation = Array(parts(0).toDouble) ++ parts(1).split(' ').filterNot(elm => elm == "").map(_.toDouble)
      if(sparse)
        LabeledPoint(observation.head, Vectors.dense(observation.tail).toSparse)
      else
        LabeledPoint(observation.head, Vectors.dense(observation.tail))
    }
  }


}
