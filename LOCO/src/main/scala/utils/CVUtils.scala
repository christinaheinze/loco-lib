package LOCO.utils

import breeze.linalg._
import scala.collection.Seq

import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import preprocessingUtils.FeatureVectorLP

import LOCO.solvers.localDual
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
   * @param response Response vector corresponding to training data set
   * @param nFeats Number of features
   * @param nPartitions Number of partitions to use to split the data matrix across columns.
   * @param nExecutors Number of executors used - needed to set the tree depth in treeReduce when
   *                   aggregating the random features.
   * @param projection Specify which projection shall be used: "sparse" for a sparse random
   *                   projection or "SDCT" for the SDCT.
   * @param useSparseStructure Set to true if sparse data structures should be used.
   * @param concatenate True if random features should be concatenated.
   *                    When set to false they are added.
   * @param nFeatsProj Dimensionality of the random projection
   * @param lambdaSeq Sequence of lambda values used in cross validation.
   * @param k Specify k for kfold cross validation.
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
      data : RDD[FeatureVectorLP],
      response: DenseVector[Double],
      nFeats : Int,
      nPartitions : Int,
      nExecutors : Int,
      projection : String,
      useSparseStructure : Boolean,
      concatenate : Boolean,
      nFeatsProj : Int,
      lambdaSeq : Seq[Double],
      k : Int,
      numIterations : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : Double = {

    // get number of observations
    val nObs = response.length

    // set seed
    util.Random.setSeed(seed)

    // create indices for training and test examples
    val times = nObs / k + 1
    val shuffledIndices = util.Random.shuffle(List.fill(times)(1 to k).flatten).take(nObs)

    // compute performance of lambda values for each training and test set
    // each training set is associated with a different RP
    // i.e. we also average over the randomness coming from the RP

    val performanceOfParams = new Array[Array[(Double, Double)]](k)
    for(fold <- 1 to k) {

      val trainingIndices = shuffledIndices.zipWithIndex.filter(x => x._1 != fold).map(_._2)
      val testIndices = shuffledIndices.zipWithIndex.filter(x => x._1 == fold).map(_._2)

      println("\nRunning for a sequence of lambda values...Fold " + fold)

      // compute LOCO coefficients for a sequence of lambda values,
      // test on local test points and return associated MSEs
      val lambdasAndErrorOnFold =
        runForLambdaSequence(
          sc, classification, seed, data, response, (trainingIndices, testIndices), nFeats,
          nPartitions, nExecutors, projection, useSparseStructure, concatenate, nFeatsProj,
          lambdaSeq, numIterations, checkDualityGap, stoppingDualityGap)

      // return performance for sequence of lambda values
      performanceOfParams(fold-1) = lambdasAndErrorOnFold
    }

    // average performance of each lambda over k train/test pairs
    val averageMSEoverKSets: Map[Double, Double] =
      performanceOfParams
        .flatten
        .groupBy(_._1)
        .mapValues(vals => vals.map(elem => elem._2))
        .mapValues(elem => breeze.linalg.sum(elem)/k.toDouble)

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
   * @param randomSeed Random seed for cross validation and choosing the examples randomly in SDCA.
   * @param data Data set
   * @param response Response vector
   * @param trainingTestIndices Tuple containing indices of training and test points
   * @param nFeats Number of features
   * @param nPartitions Number of partitions to use to split the data matrix across columns.
   * @param nExecutors Number of executors used - needed to set the tree depth in treeReduce when
   *                   aggregating the random features.
   * @param projection Specify which projection shall be used: "sparse" for a sparse random
   *                   projection or "SDCT" for the SDCT.
   * @param useSparseStructure Set to true if sparse data structures should be used.
   * @param concatenate True if random features should be concatenated.
   *                    When set to false they are added.
   * @param nFeatsProj Dimensionality of the random projection
   * @param lambdaSeq Sequence of lambda values used in cross validation.
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
   */
  def runForLambdaSequence(
      sc : SparkContext,
      classification : Boolean,
      randomSeed : Int,
      data : RDD[FeatureVectorLP],
      response: DenseVector[Double],
      trainingTestIndices : (List[Int], List[Int]),
      nFeats : Int,
      nPartitions : Int,
      nExecutors : Int,
      projection : String,
      useSparseStructure : Boolean,
      concatenate : Boolean,
      nFeatsProj : Int,
      lambdaSeq : Seq[Double],
      numIterations : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : Array[(Double, Double)] = {

    // get number of observations
    val nObs = trainingTestIndices._1.length

    // create local matrices with 1) training observations and 2) test oberservations
    val localMats: RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))] =
      preprocessing.createLocalMatrices(data, useSparseStructure, nObs, trainingTestIndices)

    // project local matrices
    val rawAndRandomFeats =
      project(localMats, projection, useSparseStructure, nFeatsProj, nObs, nFeats, randomSeed, nPartitions)

    // persist raw and random features
    rawAndRandomFeats.persist(StorageLevel.MEMORY_AND_DISK)

    // if random projection are to be concatenated, broadcast random projections
    val randomProjectionsConcatenated =
      if (concatenate) {
        sc.broadcast(
          rawAndRandomFeats
            .mapValues { case (colIndices, rawFeatsTrain, randomFeats, rawFeatsTest) => randomFeats }
            .collectAsMap())
      }
      else {
        null
      }

    // if random projection are to be added, add random projections
    val randomProjectionsAdd =
      if (!concatenate) {
        sc.broadcast(
          rawAndRandomFeats
            .values
            .map { case (colIndices, rawFeatsTrain, randomFeatures, rawFeatsTest) => randomFeatures }
            .treeReduce(_ + _, depth =
            math.max(2, math.ceil(math.log10(nExecutors) / math.log10(2))).toInt))
      }
      else {
        null
      }

    // broadcast response vector with training observations
    val responseTrainBC = sc.broadcast(response(trainingTestIndices._1).toDenseVector)

    // for each worker: compute estimates locally on design matrix with raw and random features
    val lambdasWithLocalPredictions: RDD[(Double, DenseVector[Double])] = {
      if (concatenate) {
        // extract partitionID as key, and column indices and raw features as value
        val rawFeats =
          rawAndRandomFeats
            .map { case (partitionID, (colIndices, rawFeaturesTrain, randomFeatures, rawFeaturesTest)) =>
            (partitionID, (colIndices, rawFeaturesTrain, rawFeaturesTest))
          }

        // for each local design matrix, learn coefficients
        rawFeats.flatMap { oneLocalMat =>
          localDual.runLocalDualConcatenate_lambdaSeq(
            oneLocalMat, randomProjectionsConcatenated.value, responseTrainBC.value, lambdaSeq,
            nObs, classification, numIterations, nFeatsProj, randomSeed, checkDualityGap,
            stoppingDualityGap)
        }
      } else {
        // when RPs are to be added
        rawAndRandomFeats.flatMap { oneLocalMat =>
          localDual.runLocalDualAdd_lambdaSeq(
            oneLocalMat, randomProjectionsAdd.value, responseTrainBC.value, lambdaSeq, nObs,
            classification, numIterations, nFeatsProj, randomSeed, checkDualityGap,
            stoppingDualityGap)
        }
      }
    }

    // get predicted values for each value of lambda
    val lambdasWithPredictions: RDD[(Double, DenseVector[Double])] =
      lambdasWithLocalPredictions.reduceByKey(_+_)

    // broadcast test response vector
    val responseTestBC = sc.broadcast(response(trainingTestIndices._2).toDenseVector)
    val nTest = sc.broadcast(trainingTestIndices._2.length.toDouble)

    // compute error for each lambda
    val lambdasWithErrors: Array[(Double, Double)] =
      lambdasWithPredictions.mapValues{
        predictionVector =>
          if(classification){
            sum((responseTestBC.value :* predictionVector).map(x => if(x > 0.0) 0.0 else 1.0))/nTest.value
          }else{
            math.pow(norm(responseTestBC.value - predictionVector), 2)/nTest.value
          }
      }.collect()


    // 1
    //    val lambdasWithPredictions: Map[Double, DenseVector[Double]] =
    //      lambdasWithLocalPredictions
    //        .collect()
    //        .groupBy(_._1)
    //        .map{
    //          case(lambda, localPred) =>
    //            val globalPrediction: DenseVector[Double] = localPred.map(x => x._2).reduce(_ + _)
    //            (lambda, globalPrediction)
    //      }
    //
    //    val responseTest = response(trainingTestIndices._2).toDenseVector
    //
    //    val lambdasWithErrors: Array[(Double, Double)]  =
    //      lambdasWithPredictions.mapValues{
    //        predictionVector =>
    //          if(classification){
    //            sum((responseTest :* predictionVector).map(x => if(x > 0.0) 0.0 else 1.0))/responseTest.length.toDouble
    //          }else{
    //            math.pow(norm(responseTest - predictionVector), 2)
    //          }
    //      }.toArray

    // 2
//    val lambdasWithPredictions: Array[(Double, DenseVector[Double])] =
//      lambdasWithLocalPredictions.reduceByKey(_ + _).collect()
//
//    val responseTestBC = sc.broadcast(response(trainingTestIndices._2).toDenseVector)
//
//    val lambdasWithErrors: Array[(Double, Double)] =
//      lambdasWithPredictions.map {
//        case (lambda, predictionVector) =>
//          val MSE =
//            if (classification) {
//              sum((responseTestBC.value :* predictionVector).map(x => if (x > 0.0) 0.0 else 1.0)) / responseTestBC.value.length.toDouble
//            } else {
//              math.pow(norm(responseTestBC.value - predictionVector), 2)
//            }
//        (lambda, MSE)
//      }

    rawAndRandomFeats.unpersist()

    // return lambdas with errors
    lambdasWithErrors
  }

}
