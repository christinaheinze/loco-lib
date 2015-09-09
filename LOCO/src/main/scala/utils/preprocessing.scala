package LOCO.utils

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{CSCMatrix, DenseMatrix, Vector}

import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

import preprocessingUtils.{DataPoint, FeatureVector}


object preprocessing {

  /**
   * Distribute the data over the columns and center if still needed
   * (The latter can be done with processingUtils package).
   *
   * @param trainingData Training data set.
   * @param center If true, centers features and response.
   * @param centerFeaturesOnly If true, centers features but not the response.
   * @param nWorkers Number of partitions to split the data matrix in (along columns).
   *
   * @return Return RDD containing features vectors with nWorkers partitions. In addition, return
   *         response, column means and mean response.
   */
  def distributeOverColsAndCenterRDD(
      trainingData : RDD[DataPoint],
      center : Boolean,
      centerFeaturesOnly : Boolean,
      nWorkers : Int) : (RDD[FeatureVector], Vector[Double], Vector[Double], Double) = {

    // distribute training data over columns
    val (featuresVectors, response) = distributeOverColumns(trainingData, nWorkers)

    if(center || centerFeaturesOnly) {
      // center features (so that we do not need an intercept)
      // and also return means of feature vectors
      val parsedDataByColMeansAndData : RDD[(Int, Double, FeatureVector)] =
        featuresVectors.map {
          column =>
            val n = column.observations.length
            val colMean = breeze.linalg.sum(column.observations) / n.toDouble
            (column.index, colMean,
              FeatureVector(column.index, column.observations - Vector.fill(n){colMean}))
        }.cache()

      // extract design matrix
      val parsedData_byCol =
        parsedDataByColMeansAndData
          .map{case(columnInd, colMean, featureVectors) => featureVectors}

      // collect means of feature vectors as Vector to driver
      val colMeans =
        Vector(
          parsedDataByColMeansAndData
            .map{case(columnInd, colMean, featureVectors) => (columnInd, colMean)}
            .collectAsMap().toSeq.sorted.map(_._2).toArray
        )

      // center response as well or features only
      if(center){
        val n = response.length
        val meanResponse = breeze.linalg.sum(response)/n.toDouble
        val centeredResponse = response - Vector.fill(n)(meanResponse)
        (parsedData_byCol, centeredResponse, colMeans, meanResponse)
      }
      else{
        (parsedData_byCol, response, colMeans, 0)
      }

    }
    else{
      // do not center at all (if data is already centered)
      val p = trainingData.first().features.length
      (featuresVectors, response, Vector.fill(p)(0), 0)
    }
  }

  /**
   * Takes the data set distributed over rows (RDD[DataPoint]) and distributes it over columns,
   * returning an RDD[FeatureVector]
   *
   * @param dataDistributedOverRows The data set distributed over rows.
   * @param nWorkers Number of partitions to split the data matrix (along columns).
   *
   * @return Data set distributed over columns and response vector.
   */
  def distributeOverColumns(
      dataDistributedOverRows : RDD[DataPoint],
      nWorkers : Int) : (RDD[FeatureVector], Vector[Double]) = {

    // add identifier for rows
    val dataWithRowIndices = dataDistributedOverRows.zipWithUniqueId().persist()

    // extract feature vectors
    val featuresVectors =
      dataWithRowIndices
        .map{case(dataPoint, rowInd) => (dataPoint.features, rowInd)}
        .flatMap{case((features : Vector[Double], identifier: Long)) =>
          features.toArray.zipWithIndex.zip(Array.fill(features.length)(identifier))}
        .map{case((features, colInd), rowInd) => (colInd, (rowInd, features))}
        .groupByKey(nWorkers)
        .mapValues(iter => Vector(iter.toSeq.sortBy(_._1).map(_._2).toArray))
        .map{case(colInd, features) => FeatureVector(colInd, features)}

    // extract response
    val response =
      Vector(
        dataWithRowIndices
        .map{case(dataPoint, rowInd) => (rowInd, dataPoint.label)}
        .sortByKey().map(_._2).collect()
      )

    // force featuresVectors to be computed and then unpersist dataWithRowIndices
    featuresVectors.persist(StorageLevel.MEMORY_AND_DISK).foreach(x => {})
    dataWithRowIndices.unpersist()

    (featuresVectors, response)
  }

  /**
   *  Aggregate feature vectors per partition to matrix
   *
   * @param partitionIndex ID for partition
   * @param iter Iterator over elements in partitions. The elements are of type FeatureVector.
   * @param nObs Number of observation in data set.
   *
   * @return Iterator with key-value pairs where the key is the partitiion ID and the value
   *         consists of a tuple containing the column indices and the local raw features of
   *         the partition.
   */
  def createLocalMatrices(
      partitionIndex : Int,
      iter : Iterator[FeatureVector],
      nObs : Int) : Iterator[(Int, (List[Int], DenseMatrix[Double]))] = {

    var columnIndices = new mutable.MutableList[Int]()
    var ArrayOfFeatureVecs = new ArrayBuffer[Array[Double]]()

    while(iter.hasNext) {
      val x = iter.next()
      columnIndices += x.index
      ArrayOfFeatureVecs += x.observations.toArray
    }

    val localMat = new DenseMatrix(nObs, columnIndices.length, ArrayOfFeatureVecs.toArray.flatten)

    Iterator((partitionIndex, (columnIndices.toList, localMat)))
  }

  def createLocalMatricesSparse(
                           partitionIndex : Int,
                           iter : Iterator[FeatureVector],
                           nObs : Int) : Iterator[(Int, (List[Int], CSCMatrix[Double]))] = {

    var columnIndices = new mutable.MutableList[Int]()
    val featureVectors = iter.toArray
    val builder = new CSCMatrix.Builder[Double](rows=nObs, cols=featureVectors.length)

    for(colInd <- 0 until featureVectors.length) {
      val x = featureVectors(colInd)
      columnIndices += x.index
      val featureVec = x.observations

      for(rowInd <- 0 until featureVec.length){
        builder.add(rowInd, colInd, featureVec(rowInd))
      }
    }

    val localMat = builder.result()

    Iterator((partitionIndex, (columnIndices.toList, localMat)))
  }


}
