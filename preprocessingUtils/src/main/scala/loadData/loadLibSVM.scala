package preprocessingUtils.loadData

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object loadLibSVM {

  /**
   * Reads in LIBSVM file and parses according to rows. The implementation is largely
   * identical to the one provided in MLlib.
   *
   * @param sc Spark context
   * @param path Path to text file in LIBSVM format
   * @param minPartitions Minimum number of partitions to use in resulting RDD
   * @return RDD containing the observations as arrays of doubles. The response is the respective
   *         first element in the array.
   */
  def loadLibSVMFile(
      sc: SparkContext,
      path: String,
      minPartitions: Int,
      sparse : Boolean): RDD[LabeledPoint] = {

    // parse text file
    val parsed = sc.textFile(path, minPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
      val items = line.split(' ')
      val label = items.head
      val (indices, values) = items.tail.map { item =>
        val indexAndValue = item.split(':')
        val index = indexAndValue(0).toInt
        val value = indexAndValue(1)
        (index, value)
      }.unzip
      (label, indices.toArray, values.toArray)
    }

    // Determine number of features.
    val d = {
      parsed.persist(StorageLevel.MEMORY_ONLY_SER)
      parsed.map { case (label, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max)
    }

    parsed.map { case (label, indices, values) =>
      val features = Array.fill[String](d)("0")
      for (a <- 0 until indices.length) {
        features(indices(a) - 1) = values(a)
      }
      if(sparse)
        LabeledPoint(label.toDouble, Vectors.dense(features.map(_.toDouble)).toSparse)
      else
        LabeledPoint(label.toDouble, Vectors.dense(features.map(_.toDouble)))
    }
  }

}
