package LOCO.solvers

import breeze.linalg.Vector
import scala.collection._

import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import preprocessingUtils.DataPoint
import LOCO.utils.preprocessing
import LOCO.utils.ProjectionUtils._


object runLOCO {

  /**
   * Runs LOCO
   *
   * @param sc Spark Context
   * @param classification True if the problem at hand is a classification problem. In this case
   *                       an l2-penalized SVM is trained. When set to false ridge regression is
   *                       used.
   * @param myseed Random seed for cross validation and choosing the examples randomly in SDCA.
   * @param trainingData Training data set
   * @param center True if both the features and the response should be centered.
   * @param centerFeaturesOnly True if only the features but not the response should be centered.
   * @param nPartitions Number of partitions to use to split the data matrix across columns.
   * @param nExecutors Number of executors used - needed to set the tree depth in treeReduce when
   *                   aggregating the random features.
   * @param projection Specify which projection shall be used: "sparse" for a sparse
   *                   random projection or "SRHT" for the SRHT. Note that the latter is not
   *                   threadsafe!
   * @param concatenate True if random features should be concatenated.
   *                    When set to false they are added.
   * @param nFeatsProj Dimensionality of the random projection
   * @param lambda Regularization paramter to use in algorithm
   * @param CVKind Can be "global", "local", or "none". If global, the errors are evaluated globally
   *               with the coefficients returned by LOCO. If local, the errors are evaluated on
   *               the local design matrices of the individual workers. I.e. this option does not
   *               require additional communication.
   * @param lambdaSeq Sequence of lambda values used in cross validation.
   * @param kfold Specify k for kfold cross validation.
   * @param optimizer Can be "SDCA" for stochastic dual coordinate ascent or "factorie" to use
   *                  batch solvers provided in the factorie library
   *                  (http://factorie.cs.umass.edu/).
   * @param numIterations Specify number of iterations used in SDCA.
   * @param checkDualityGap If the optimizer is SDCA, specify whether the duality gap should be
   *                        computer after each iteration. Note that this is expensive as it
   *                        requires a pass over the entire (local) data set. Should only be used
   *                        for tuning purposes.
   * @param stoppingDualityGap If the optimizer is SDCA, specify the size of the duality gap at
   *                           which the optimization should end. If it is not reached after
   *                           numIterations, the optimization ends nonetheless.
   *
   * @return Return the coefficients estimated by LOCO, the time stamp when timing was started,
   *         the column means and the mean response (to center the test set later).
   */
  def run(
      sc : SparkContext,
      classification : Boolean,
      myseed : Int,
      trainingData : RDD[DataPoint],
      center : Boolean,
      centerFeaturesOnly : Boolean,
      nPartitions : Int,
      nExecutors : Int,
      projection : String,
      concatenate : Boolean,
      nFeatsProj : Int,
      lambda : Double,
      CVKind : String,
      lambdaSeq : Seq[Double],
      kfold : Int,
      optimizer : String,
      numIterations : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : (Vector[Double], Long, Vector[Double], Double) = {

    // distribute training data over columns, center data (so that we do not need an intercept)
    // and also return means of response and feature vectors (to driver)
    val (parsedDataByCol, response, colMeans, meanResponse) =
      preprocessing.distributeOverColsAndCenterRDD(
        trainingData, center, centerFeaturesOnly, nPartitions)

    // get number of observations and number of features
    val nObs = response.length
    val nFeats = colMeans.length

    println("\nNumber of observations: " + nObs)
    println("\nNumber of features: " + nFeats)
    println("\nApprox. number of raw features per worker: " + nFeats / nPartitions.toDouble)
    println("\nProjection dimension: " + nFeatsProj)
    println("\nPartitions training over rows: " + trainingData.partitions.size)
    println("\nPartitions training over cols: " + parsedDataByCol.partitions.size)

    // start timing
    val t1 = System.currentTimeMillis

    // project local matrices
    val rawAndRandomFeats =
      project(parsedDataByCol, projection, concatenate, nFeatsProj, nObs, nFeats,
        myseed, nPartitions)

    // force evaluation of rawAndRandomFeats RDD and unpersist parsedDataByCol
    // (only needed for timing purposes)
    rawAndRandomFeats.persist(StorageLevel.MEMORY_AND_DISK).foreach(x => {})
    parsedDataByCol.unpersist()

    // if random projection are to be concatenated, broadcast random projections
    val randomProjectionsConcatenated =
      if(concatenate){
        sc.broadcast(
          rawAndRandomFeats
            .mapValues{case(colIndices, rawFeats, randomFeats) => randomFeats}
            .collectAsMap())
      }else{
        null
      }

    // if random projection are to be added, add random projections
    val randomProjectionsAdded =
      if(!concatenate){
        sc.broadcast(
          rawAndRandomFeats
            .values
            .map{case(colIndices, rawFeatures, randomFeatures) => randomFeatures}
            .treeReduce(_ + _, depth =
              math.max(2, math.ceil(math.log10(nExecutors)/math.log10(2))).toInt))
      }else{
        null
      }

    // broadcast response vector
    val responseBroadcast = sc.broadcast(response)

    // set flag if local CV was chosen
    val doLocalCV = if(CVKind == "local") true else false

    // for each worker: compute estimates locally on design matrix with raw and random features
    val betaLocoAsMap = {
      if (concatenate){
        // extract partitionID as key, and column indices and raw features as value
        val rawFeats =
          rawAndRandomFeats
            .map{case(partitionID, (colIndices, rawFeatures, randomFeatures)) =>
              (partitionID, (colIndices, rawFeatures))}

        // for each local design matrix, learn coefficients
        rawFeats.map{ oneLocalMat =>
          // regression with factorie
          if(!classification && optimizer == "factorie")
            localRidge.runLocalRidgeRegressionConcatenate(
              oneLocalMat, randomProjectionsConcatenated.value, responseBroadcast.value,
              concatenate, doLocalCV, kfold, lambdaSeq, lambda, nObs, nFeatsProj)
          else
            localDual.runLocalDualConcatenate(
              oneLocalMat, randomProjectionsConcatenated.value, responseBroadcast.value,
              doLocalCV, kfold, lambdaSeq, lambda, nObs, classification, optimizer,
              numIterations, nFeatsProj, checkDualityGap, stoppingDualityGap)
        }
      }else{
        // when RPs are to be added
        rawAndRandomFeats.map{ oneLocalMat =>
          // regression with factorie
          if(!classification && optimizer == "factorie")
              localRidge.runLocalRidgeRegressionAdd(
                oneLocalMat, randomProjectionsAdded.value, responseBroadcast.value,
                 concatenate, doLocalCV, kfold, lambdaSeq, lambda, nObs, nFeatsProj)
            else
              localDual.runLocalDualAdd(
                oneLocalMat, randomProjectionsAdded.value, responseBroadcast.value,
                doLocalCV, kfold, lambdaSeq, lambda, nObs, classification, optimizer,
                numIterations, nFeatsProj, checkDualityGap, stoppingDualityGap)
        }
      }
    }.flatMap{case(colIndices, coefficients) => colIndices.zip(coefficients.toArray)}.collectAsMap()

    // unpersist raw and random features
    rawAndRandomFeats.unpersist()

    // sort coefficients by column index and return LOCO coefficients, starting time stamp,
    // column means and the mean of the response
    (Vector(betaLocoAsMap.toSeq.sorted.map(_._2).toArray), t1, colMeans, meanResponse)
  }
}
