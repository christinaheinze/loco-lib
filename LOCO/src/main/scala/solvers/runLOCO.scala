package LOCO.solvers

import breeze.linalg._
import org.apache.spark.broadcast.Broadcast
import scala.collection._

import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


import preprocessingUtils.FeatureVectorLP
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
   * @param randomSeed Random seed for random projections and choosing the examples randomly in SDCA.
   * @param trainingDataByCol Training data set
   * @param nPartitions Number of partitions to use to split the data matrix across columns.
   * @param nExecutors Number of executors used - needed to set the tree depth in treeReduce when
   *                   aggregating the random features.
   * @param projection Specify which projection shall be used: "sparse" for a sparse
   *                   random projection or "SDCT" for the SDCT.
   * @param concatenate True if random features should be concatenated.
   *                    When set to false they are added.
   * @param nFeatsProj Dimensionality of the random projection
   * @param lambda Regularization paramter to use in algorithm
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
      randomSeed : Int,
      trainingDataByCol : RDD[FeatureVectorLP],
      response : DenseVector[Double],
      nFeats : Int,
      nPartitions : Int,
      nExecutors : Int,
      projection : String,
      useSparseStructure : Boolean,
      concatenate : Boolean,
      nFeatsProj : Int,
      lambda : Double,
      numIterations : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double,
      privateLOCO : Boolean,
      privateEps : Double,
      privateDelta : Double) : (DenseVector[Double], Long, Long, Long) = {

    // get number of observations
    val nObs = response.size

    //
    val naive = if(nFeatsProj == 0) true else false


    println("\nNumber of observations: " + nObs)
    println("\nNumber of features: " + nFeats)
    println("\nApprox. number of raw features per worker: " +
      nFeats / trainingDataByCol.partitions.size.toDouble)
    println("\nProjection dimension: " + nFeatsProj)
    println("\nPartitions training over cols: " + trainingDataByCol.partitions.size)

    // create local matrices from feature vectors
    val localMats = preprocessing.createLocalMatrices(trainingDataByCol, useSparseStructure, nObs, null)

    // persist local matrices and unpersist training data RDD
    localMats.persist(StorageLevel.MEMORY_AND_DISK).foreach(x => {})
    trainingDataByCol.unpersist()

    // start timing of LOCO
    val t1 = System.currentTimeMillis

    // project local matrices
    val rawAndRandomFeats : RDD[(Int, (List[Int], Matrix[Double], DenseMatrix[Double], Option[Matrix[Double]]))] =
      project(localMats, projection, useSparseStructure, nFeatsProj, nObs, nFeats, randomSeed,
        nPartitions, privateLOCO, privateEps, privateDelta)

    // force evaluation of rawAndRandomFeats RDD and unpersist localMats (only needed for timing purposes)
    rawAndRandomFeats.persist(StorageLevel.MEMORY_AND_DISK).foreach(x => {})
    localMats.unpersist()

    // time: random features have been computed
    val tRPComputed = System.currentTimeMillis

    // if random projection are to be concatenated, broadcast random projections
    val randomProjectionsConcatenated: Broadcast[Map[Int, DenseMatrix[Double]]] =
      if(concatenate){
        sc.broadcast(
          rawAndRandomFeats
            .mapValues{case(colIndices, rawFeatsTrain, randomFeats, rawFeatsTest) => randomFeats}
            .collectAsMap()
        )
      }else{
        null
      }

    // if random projection are to be added, add random projections and broadcast
    val randomProjectionsAdded: Broadcast[DenseMatrix[Double]] =
      if(!concatenate){
        sc.broadcast(
          rawAndRandomFeats
            .values
            .map{case(colIndices, rawFeatsTrain, randomFeats, rawFeatsTest) => randomFeats}
            .treeReduce(_ + _, depth =
              math.max(2, math.ceil(math.log10(nExecutors)/math.log10(2))).toInt))
      }else{
        null
      }

    // time: random features have been communicated
    val tRPCommunicated = System.currentTimeMillis

    // broadcast response vector
    val responseBroadcast = sc.broadcast(response)

    // for each worker: compute estimates locally on design matrix with raw and random features
    val betaLocoAsMap: Map[Int, Double] = {
      if (concatenate){
        // extract partitionID as key, and column indices and raw features as value
        val rawFeatsTrainRDD =
          rawAndRandomFeats
            .map{case(partitionID, (colIndices, rawFeatsTrain, randomFeatures, rawFeatsTest)) =>
              (partitionID, (colIndices, rawFeatsTrain))
            }

        // for each local design matrix, learn coefficients
        rawFeatsTrainRDD.map{ oneLocalMat =>
            localDual.runLocalDualConcatenate(
              oneLocalMat, randomProjectionsConcatenated.value, responseBroadcast.value, lambda,
              nObs, classification, numIterations, nFeatsProj, randomSeed, checkDualityGap,
              stoppingDualityGap, naive)
        }
      }else{
        // when RPs are to be added
        rawAndRandomFeats.map{ oneLocalMat =>
              localDual.runLocalDualAdd(
                oneLocalMat, randomProjectionsAdded.value, responseBroadcast.value, lambda, nObs,
                classification, numIterations, nFeatsProj, randomSeed, checkDualityGap,
                stoppingDualityGap, naive)
        }
      }
    }.flatMap{case(colIndices, coefficients) => colIndices.zip(coefficients.toArray)}.collectAsMap()

    // unpersist raw and random features
    rawAndRandomFeats.unpersist()

    val betaLoco = DenseVector.fill(nFeats)(0.0)
    for(entry <- 0 until nFeats){
        betaLoco(entry) = betaLocoAsMap.getOrElse(entry, 0.0)
    }

    // sort coefficients by column index and return LOCO coefficients and time stamps
    (betaLoco, t1, tRPComputed, tRPCommunicated)
  }
}