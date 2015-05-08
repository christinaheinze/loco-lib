package LOCO.utils

import breeze.linalg.{DenseVector, Vector}
import scala.collection.Seq

import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils

import preprocessingUtils.DataPoint
import preprocessingUtils.utils.metrics

import LOCO.solvers.{localDual, localRidge}
import LOCO.utils.ProjectionUtils._


object CVUtils {

  /**
   * Runs cross validation for LOCO, targeting the global prediction error
   *
   * @param sc Spark Context
   * @param classification True if the problem at hand is a classification problem. In this case
   *                       an l2-penalized SVM is trained. When set to false ridge regression is
   *                       used.
   * @param seed Random seed for cross validation and choosing the examples randomly in SDCA.
   * @param data Training data set
   * @param center True if both the features and the response should be centered.
   * @param centerFeaturesOnly True if only the features but not the response should be centered.
   * @param nPartitions Number of partitions to use to split the data matrix across columns.
   * @param nExecutors Number of executors used - needed to set the tree depth in treeReduce when
   *                   aggregating the random features.
   * @param projection Specify which projection shall be used: "sparse" for a sparse random
   *                   projection or "SDCT" for the SDCT. Note that the latter is not threadsafe!
   * @param flagFFTW flag for SDCT/FFTW: 64 corresponds to FFTW_ESTIMATE,
   *                 0 corresponds to FFTW_MEASURE
   * @param concatenate True if random features should be concatenated.
   *                    When set to false they are added.
   * @param nFeatsProj Dimensionality of the random projection
   * @param lambdaSeq Sequence of lambda values used in cross validation.
   * @param k Specify k for kfold cross validation.
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
   * @return Lambda value minimizing the prediction error found by cross validation.
   */
  def globalCV(
      sc : SparkContext,
      classification : Boolean,
      seed : Int,
      data : RDD[DataPoint],
      center : Boolean,
      centerFeaturesOnly : Boolean,
      nPartitions : Int,
      nExecutors : Int,
      projection : String,
      flagFFTW : Int,
      concatenate : Boolean,
      nFeatsProj : Int,
      lambdaSeq : Seq[Double],
      k : Int,
      optimizer : String,
      numIterations : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : Double = {

    // create k training and test set pairs
    val kTrainingAndTestSets =
      MLUtils.kFold(data, k, seed).map{case (train, test) =>
        (train.persist(StorageLevel.MEMORY_ONLY_SER), test.persist(StorageLevel.MEMORY_ONLY_SER))
      }

    // compute performance of lambda values for each training and test set
    // each training set is associated with a different RP
    // i.e. we also average over the randomness coming from the RP
    val performanceOfParams : Array[(Double, Double)] =
      kTrainingAndTestSets.flatMap {

        case (training, test) =>
          // for each training and test set, compute LOCO coefficients for
          // a sequence of lambda values
          val (lambdasAndCoefficientVectorsMap, colMeans, meanResponse) =
            runForLambdaSequence(
              sc, classification, seed, training, center, centerFeaturesOnly, nPartitions,
              nExecutors, projection, flagFFTW, concatenate, nFeatsProj, lambdaSeq, optimizer,
              numIterations, checkDualityGap, stoppingDualityGap)

          // collect lambda with coefficients as map
//          val lambdasAndCoefficientVectorsMap = lambdasAndCoefficientVectors.collectAsMap()

          // broadcast column means if features should be centered
          val colMeansBroadcast = if(center || centerFeaturesOnly) sc.broadcast(colMeans) else null

          // center test data by row if needed
          val testCentered = test.map{ elem =>
            if(center){
              DataPoint(elem.label - meanResponse, elem.features - colMeansBroadcast.value)
            }else if(centerFeaturesOnly){
              DataPoint(elem.label, elem.features - colMeansBroadcast.value)
            }
            else{
              elem
            }
          }.persist(StorageLevel.MEMORY_ONLY_SER)

          // compute MSEs
          val performance =
            lambdasAndCoefficientVectorsMap.mapValues{coefficients =>
              if(classification){
                metrics.computeClassificationError(coefficients, testCentered)(x =>
                  (x.label, x.features))
              }
            else{
                metrics.compute_MSE(coefficients, testCentered)(x => (x.label, x.features))
              }
            }

          // unpersist test set
          testCentered.unpersist()

          // return performance for sequence of lambda values
          performance
    }

    kTrainingAndTestSets.map{case (train, test) =>
      (train.unpersist(), test.unpersist())
    }

    // average performance of each lambda over k train/test pairs
    val averageMSEoverKSets =
      performanceOfParams
        .groupBy(_._1).mapValues(vals => vals.map(elem => elem._2))
        .mapValues(elem => breeze.linalg.sum(elem)/elem.length.toDouble)

    // find min test MSE
    val bestMSE = averageMSEoverKSets.minBy(_._2)

    // find corresponding parameters
    println("Global CV: Best lambda is " + bestMSE._1 + " with error " + bestMSE._2)

    // return lambda value associated with the best MSE
    bestMSE._1
  }

  /**
   * Runs LOCO for a sequence of lambda values
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
   * @param projection Specify which projection shall be used: "sparse" for a sparse random
   *                   projection or "SDCT" for the SDCT. Note that the latter is not threadsafe!
   * @param flagFFTW flag for SDCT/FFTW: 64 corresponds to FFTW_ESTIMATE,
   *                 0 corresponds to FFTW_MEASURE
   * @param concatenate True if random features should be concatenated.
   *                    When set to false they are added.
   * @param nFeatsProj Dimensionality of the random projection
   * @param lambdaSeq Sequence of lambda values used in cross validation.
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
   * @return Returns an RDD containing the lambda values with the corresponding coefficient vectors,
   *         as well as the column means and the mean response for the given training data.
   */ //TODO code here is partially duplicated from runLOCO
  def runForLambdaSequence(
      sc : SparkContext,
      classification : Boolean,
      myseed : Int,
      trainingData : RDD[DataPoint],
      center : Boolean,
      centerFeaturesOnly : Boolean,
      nPartitions : Int,
      nExecutors : Int,
      projection : String,
      flagFFTW : Int,
      concatenate : Boolean,
      nFeatsProj : Int,
      lambdaSeq : Seq[Double],
      optimizer : String,
      numIterations : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : (scala.collection.Map[Double, Vector[Double]], Vector[Double], Double) = {

    // distribute training data over columns, center data (so that we do not need an intercept)
    // and also return means of response and feature vectors (to driver)
    val (parsedDataByCol, response, colMeans, meanResponse) =
      preprocessing.distributeOverColsAndCenterRDD(
        trainingData, center, centerFeaturesOnly, nPartitions)

    // get number of observations and number of features
    val nObs = response.length
    val nFeats = colMeans.length

    println("\nRunning for a sequence of lambda values...")
    println("\nNumber of observations: " + nObs)
    println("\nNumber of features: " + nFeats)
    println("\nApprox. number of raw features per worker: " + nFeats / nPartitions.toDouble)
    println("\nProjection dimension: " + nFeatsProj)
    println("\nPartitions training over cols: " + parsedDataByCol.partitions.size)

    // project local matrices
    val rawAndRandomFeats =
      project(parsedDataByCol, projection, flagFFTW, concatenate, nFeatsProj, nObs, nFeats,
        myseed, nPartitions)

    rawAndRandomFeats.persist(StorageLevel.MEMORY_AND_DISK_SER)

    // if random projection are to be concatenated, broadcast random projections
    val randomProjectionsConcatenated =
      if(concatenate){
        sc.broadcast(
          rawAndRandomFeats
            .mapValues{case(colIndices, rawFeats, randomFeats) => randomFeats}
            .collectAsMap())
      }
      else{
        null
      }

    // if random projection are to be added, add random projections
    val randomProjectionsAdd =
      if(!concatenate)
        sc.broadcast(
          rawAndRandomFeats
            .values
            .map{case(colIndices, rawFeatures, randomFeatures) => randomFeatures}
            .treeReduce(_ + _, depth =
              math.max(2, math.ceil(math.log10(nExecutors)/math.log10(2))).toInt))
      else
        null

    // broadcast response vector
    val responseBroadcast = sc.broadcast(response)

    // for each worker: compute estimates locally on design matrix with raw and random features
    val betaLocoWithLambdas : RDD[Seq[(Double, (List[Int], Vector[Double]))]] = {
      if (concatenate){
        // extract partitionID as key, and column indices and raw features as value
        val rawFeats =
          rawAndRandomFeats
            .map{case(partitionID, (colIndices, rawFeatures, randomFeatures)) =>
              (partitionID, (colIndices, rawFeatures))}

        // for each local design matrix, learn coefficients
        rawFeats.map{ oneLocalMat =>
          // regression with factorie
          if (!classification && optimizer == "factorie")
            localRidge.runLocalRidgeRegressionConcatenate_lambdaSeq(
              oneLocalMat, randomProjectionsConcatenated.value, responseBroadcast.value,
              concatenate,  lambdaSeq, nObs, nFeatsProj)
          else
            localDual.runLocalDualConcatenate_lambdaSeq(
              oneLocalMat, randomProjectionsConcatenated.value, responseBroadcast.value,
              lambdaSeq, nObs, myseed, classification, optimizer,
              numIterations, nFeatsProj, checkDualityGap, stoppingDualityGap)
        }
      }else{
        // when RPs are to be added
        rawAndRandomFeats.map{ oneLocalMat =>
          // regression with factorie
          if(!classification && optimizer == "factorie")
              localRidge.runLocalRidgeRegressionAdd_lambdaSeq(
                oneLocalMat, randomProjectionsAdd.value, responseBroadcast.value,
                concatenate, lambdaSeq, nObs, nFeatsProj)
            else
              localDual.runLocalDualAdd_lambdaSeq(
                oneLocalMat, randomProjectionsAdd.value, responseBroadcast.value,
                lambdaSeq, nObs, myseed, classification, optimizer,
                numIterations, nFeatsProj, checkDualityGap, stoppingDualityGap)
          }
      }
    }

    // extract lambdas with coefficient vectors
    val lambdasAndCoefficientVectors =
      betaLocoWithLambdas
        .flatMap(x => x.iterator)
        .groupBy(_._1)
        .mapValues{x =>
          val listWithIndAndCoef =
            x.map(elem => elem._2)
              .flatMap(resultTuple => resultTuple._1.zip(resultTuple._2.toArray))
          listWithIndAndCoef.toMap
        }
        .mapValues(x => Vector(x.toSeq.sorted.map(_._2).toArray))
        .collectAsMap()

    rawAndRandomFeats.unpersist()

    // return lambdas with coefficient vectors, column means and mean of response
    (lambdasAndCoefficientVectors, colMeans, meanResponse)
  }

}
