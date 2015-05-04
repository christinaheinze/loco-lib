package preprocessingUtils.loadData

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


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
      minPartitions: Int): RDD[Array[Double]] = {

    // load data
    val data = sc.textFile(path, minPartitions)

    // map each element to Array[Double]
    data.map { line =>
      line.split(' ').filterNot(elm => elm == "").map(_.toDouble)
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
      minPartitions: Int): RDD[Array[Double]] = {

    // load data
    val data = sc.textFile(path, minPartitions)

    // map each element to Array[Double]
    data.map { line =>
      val parts = line.split(',')
      Array(parts(0).toDouble) ++ parts(1).split(' ').filterNot(elm => elm == "").map(_.toDouble)
    }
  }


}
