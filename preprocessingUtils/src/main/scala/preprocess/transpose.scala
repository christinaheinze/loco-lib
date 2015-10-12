package preprocessingUtils.preprocess

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import preprocessingUtils.FeatureVectorLP


object transpose {
  /**
   * Transposes an RDD of LabeledPoints to contain an RDD of FeatureVectorLPs, so that data is
   * distributed across workers according to the features instead of the observations.
   *
   * @param dataDistributedOverRows RDD of LabeledPoints
   * @param nWorkers Number of partitions
   * @param sparse Set to true, if sparse data structures should be used.
   * @return RDD of FeatureVectorLPs
   */
  def distributeOverColumns(dataDistributedOverRows : RDD[LabeledPoint],
                             nWorkers : Int,
                             sparse : Boolean) : (RDD[FeatureVectorLP], Vector, Int) = {

    // add identifier for rows
    val dataWithRowIndices = dataDistributedOverRows.zipWithUniqueId().persist()

    // extract number of features
    val nFeats = dataDistributedOverRows.first().features.size

    // extract response
    val response =
      Vectors.dense(
        dataWithRowIndices
          .map{case(dataPoint, rowInd) => (rowInd, dataPoint.label)}
          .sortByKey().map(_._2).collect()
      )

    val nObs = response.size

    // extract feature vectors
    val featuresVectors =
      if(sparse)
        dataWithRowIndices
          .map{case(dataPoint, rowInd) => (dataPoint.features, rowInd)}
          .flatMap{case((features : Vector, rowIndex: Long)) =>
            val sparseFeatures = features.toSparse
            val numActives = sparseFeatures.numActives
            val colInds = sparseFeatures.indices
            val values = sparseFeatures.values
          values.zip(colInds).zip(Array.fill(numActives)(rowIndex))}
          .map{case((features, colInd), rowInd) => (colInd, (rowInd.toInt, features))}
          .groupByKey()
          .mapValues{iter =>
            val tempSeq = iter.toSeq.sortBy(_._1)
            Vectors.sparse(
              size = nObs,
              indices = tempSeq.map(x => x._1).toArray,
              values = tempSeq.map(x => x._2).toArray)
        }.map{case(colInd, features) => FeatureVectorLP(colInd, features)}
      else
        dataWithRowIndices
          .map{case(dataPoint, rowInd) => (dataPoint.features, rowInd)}
          .flatMap{case((features : Vector, rowIndex: Long)) =>
          features.toArray.zipWithIndex.zip(Array.fill(features.size)(rowIndex))}
          .map{case((features, colInd), rowInd) => (colInd, (rowInd, features))}
          .groupByKey()
          .mapValues{iter => Vectors.dense(iter.toSeq.sortBy(_._1).map(_._2).toArray)
        }.map{case(colInd, features) => FeatureVectorLP(colInd, features)}

    // force featuresVectors to be computed and then unpersist dataWithRowIndices
    featuresVectors.persist(StorageLevel.MEMORY_AND_DISK).foreach(x => {})
    dataWithRowIndices.unpersist()

    (featuresVectors, response, nFeats)
  }
}
