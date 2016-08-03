package LOCO.utils

import breeze.linalg._
import breeze.numerics._
import scala.collection.Seq

import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import preprocessingUtils.FeatureVectorLP

import LOCO.solvers.{SDCA, localDual}
import LOCO.utils.ProjectionUtils._

import scala.collection.immutable.Iterable

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
      logistic : Boolean,
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
      stoppingDualityGap : Double,
      privateLOCO : Boolean,
      privateEps : Double,
      privateDelta : Double,
      debug : Boolean) : (Double, Option[Array[(Double, Double, Double)]]) = {

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
          sc, classification, logistic, seed, data, response, (trainingIndices, testIndices), nFeats,
          nPartitions, nExecutors, projection, useSparseStructure, concatenate, nFeatsProj,
          lambdaSeq, numIterations, checkDualityGap, stoppingDualityGap,
          privateLOCO, privateEps, privateDelta)

      // return performance for sequence of lambda values
      performanceOfParams(fold-1) = lambdasAndErrorOnFold
    }

    // average performance of each lambda over k train/test pairs
    val lambdaWithErrors: Map[Double, DenseVector[Double]] =
      performanceOfParams
        .flatten
        .groupBy(_._1)
        .mapValues(vals => vals.map(elem => elem._2))
        .mapValues(arrays => DenseVector(arrays))

    val averageMSEoverKSets: Map[Double, Double] =
      lambdaWithErrors.mapValues(elem => sum(elem)/elem.length.toDouble)

    println("Performance: " + averageMSEoverKSets.toSeq.sortBy(_._2))

    val lambda_with_mse_stats: Option[Array[(Double, Double, Double)]] =
      if(debug){
        val lambda_with_mse_sd = lambdaWithErrors.map{ case(lambda : Double, elem : DenseVector[Double]) =>
          val mean = sum(elem)/elem.length.toDouble
          val sumOfSq = sum(elem.map(arrayElement => math.pow(arrayElement - mean, 2)))
          val sd: Double = 1/(elem.length.toDouble - 1) * sumOfSq
          (lambda, mean, sd)
        }.toArray
        Some(lambda_with_mse_sd)
      }else{
        None
      }


    // find min test MSE
    val bestMSE = averageMSEoverKSets.minBy(_._2)

    // find corresponding parameters
    println("Global CV: Best lambda is " + bestMSE._1 + " with error " + bestMSE._2)

    // return lambda value associated with the best MSE
    (bestMSE._1, lambda_with_mse_stats)
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
      logistic : Boolean,
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
      stoppingDualityGap : Double,
      privateLOCO : Boolean,
      privateEps : Double,
      privateDelta : Double) : Array[(Double, Double)] = {

    // get number of observations
    val nObs = trainingTestIndices._1.length
    val naive = if(nFeatsProj == 0) true else false


    // create local matrices with 1) training observations and 2) test oberservations
    val localMats: RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))] =
      preprocessing.createLocalMatrices(data, useSparseStructure, nObs, trainingTestIndices)

    // project local matrices
    val rawAndRandomFeats =
      project(localMats, projection, useSparseStructure, nFeatsProj, nObs, nFeats, randomSeed,
        nPartitions, privateLOCO, privateEps, privateDelta)

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
            nObs, classification, logistic, numIterations, nFeatsProj, randomSeed, checkDualityGap,
            stoppingDualityGap, naive)
        }
      } else {
        // when RPs are to be added
        rawAndRandomFeats.flatMap { oneLocalMat =>
          localDual.runLocalDualAdd_lambdaSeq(
            oneLocalMat, randomProjectionsAdd.value, responseTrainBC.value, lambdaSeq, nObs,
            classification, logistic, numIterations, nFeatsProj, randomSeed, checkDualityGap,
            stoppingDualityGap, naive)
        }
      }
    }

    val lambdasWithLocalPreds = lambdasWithLocalPredictions
//      if(privateLOCO){
//        val r = new scala.util.Random()
//        val noiseVector =  DenseVector.tabulate(trainingTestIndices._2.length){case (i) => 0.95*r.nextGaussian()}
//        lambdasWithLocalPredictions.mapValues(predictions => predictions + noiseVector)
//      }else{
//        lambdasWithLocalPredictions
//      }

    // get predicted values for each value of lambda
    val lambdasWithPredictions: RDD[(Double, DenseVector[Double])] =
      lambdasWithLocalPreds.reduceByKey(_+_)

    // broadcast test response vector
    val responseTestBC = sc.broadcast(response(trainingTestIndices._2).toDenseVector)
    val nTest = sc.broadcast(trainingTestIndices._2.length.toDouble)

    // compute error for each lambda
    val lambdasWithErrors: Array[(Double, Double)] =
      lambdasWithPredictions.mapValues{
        predictionVector =>
          if(classification){
            if(logistic){
              val probabilities = predictionVector.map(x => 1/(1+math.exp(-x)))
              val mappedToOutcome: DenseVector[Double] = probabilities.map(x => if(x > 0.5) 1.0 else -1.0)
              0.5*sum(abs(mappedToOutcome - responseTestBC.value))/nTest.value
            }else{
              sum((responseTestBC.value :* predictionVector).map(x => if(x > 0.0) 0.0 else 1.0))/nTest.value
            }

          }else{
            math.pow(norm(responseTestBC.value - predictionVector), 2)/nTest.value
          }
      }.collect()


    rawAndRandomFeats.unpersist()

    // return lambdas with errors
    lambdasWithErrors
  }


  def crossValidationDualLocal(
                                designMat : DenseMatrix[Double],
                                response : Vector[Double],
                                lambdaSeq : Seq[Double],
                                k : Int,
                                myseed : Int,
                                nObs : Int,
                                numIterations : Int,
                                numFeatures : Int,
                                classification : Boolean,
//                                logistic : Boolean,
                                checkDualityGap : Boolean,
                                stoppingDualityGap : Double,
                                numRawFeatures : Int) : (Double, Option[Array[(Double, Double, Double)]]) = {

    // set seed
    util.Random.setSeed(myseed)
    val times = nObs/k + 1

    // create indices for training and test examples
    val shuffledIndices = util.Random.shuffle(List.fill(times)(1 to k).flatten).take(nObs)

    // create training and test sets
    val trainingAndTestSets: Array[((DenseMatrix[Double], DenseVector[Double]), (DenseMatrix[Double], DenseVector[Double]))] =
      (1 to k).map { fold =>

        val testInd = shuffledIndices.zipWithIndex.filter(x => x._1 == fold).map(_._2)
        val myTestExamples =
          (designMat(testInd ,::).toDenseMatrix, response(testInd).toDenseVector)

        val trainInd = shuffledIndices.zipWithIndex.filter(x => x._1 != fold).map(_._2)
        val myTrainingExamples =
          (designMat(trainInd ,::).toDenseMatrix, response(trainInd).toDenseVector)

        (myTrainingExamples, myTestExamples)
      }.toArray

    // for each lambda in sequence, return average MSE
    val k_mse = trainingAndTestSets.flatMap {
      // for each train and test set pair...
      case (train, test) =>

        val n = train._1.rows

        // for each lambda...
        lambdaSeq.map{ currentLambda =>
          // ... train on train and test on test
          val alpha =
              if(classification){
                SDCA.localSDCA(
                  train._1, train._2, numIterations, currentLambda, n, numFeatures, myseed,
                  checkDualityGap, stoppingDualityGap)
              }else{
                SDCA.localSDCARidge(
                  train._1, train._2, numIterations, currentLambda, n, numFeatures, myseed,
                  checkDualityGap, stoppingDualityGap)
              }

          val primalVarsNotScaled : DenseMatrix[Double] =
            train._1.t * new DenseMatrix(n, 1, alpha.toArray)
          val scaling = 1.0/(n*currentLambda)
          val beta_hat_k : DenseVector[Double] = primalVarsNotScaled.toDenseVector * scaling

          // compute performance
          val performance =
            if(classification){
//
//              if(logistic){
//
//              }else{
                // compute predictions
                val predictions : DenseVector[Double] =
                  test._2 :*
                    (test._1 * new DenseMatrix(numFeatures, 1, beta_hat_k.toArray)).toDenseVector

                // compute misclassification error
                predictions
                  .toArray
                  .map(x => if(x > 0.0) 0.0 else 1.0)
                  .sum/test._2.length.toDouble
//              }

            }
            else{
              // compute predictions
              val predictions : DenseVector[Double] =
                (test._1 * new DenseMatrix(numFeatures, 1, beta_hat_k.toArray)).toDenseVector
              //                (test._1(::,0 until numRawFeatures).toDenseMatrix * new DenseMatrix(numRawFeatures, 1, beta_hat_k(0 until numRawFeatures).toArray)).toDenseVector

              // compute MSE
//              1/test._2.length.toDouble * math.pow(norm(test._2 - predictions), 2)

              math.pow(norm(test._2 - predictions), 2)/math.pow(norm(test._2 - breeze.stats.mean(test._2)), 2)

            }

          //return lambda and the performance for this lambda and this train/test pair
          (currentLambda, performance)
        }
    }

    // average performance of each lambda over k train/test pairs
    val lambda_with_errors: Map[Double, DenseVector[Double]] =
      k_mse
        .groupBy(_._1)
        .mapValues(vals => vals.map(elem => elem._2))
        .mapValues(arrays => DenseVector(arrays))

    val lambda_with_mse: Map[Double, Double] =
      lambda_with_errors.mapValues(elem => breeze.stats.mean(elem)) // sum(elem)/elem.length.toDouble)

    val debug = true

    val lambda_with_mse_stats: Option[Array[(Double, Double, Double)]] =
      if(debug){
        val lambda_with_mse_sd = lambda_with_errors.map{ case(lambda : Double, elem : DenseVector[Double]) =>
          val mean : Double =  breeze.stats.mean(elem) //sum(elem)/elem.length.toDouble
//          val sumOfSq = breeze.stats.stddev(elem) // sum(elem.map(arrayElement => math.pow(arrayElement - mean, 2)))
          val sd: Double =  breeze.stats.stddev(elem) // //1/(elem.length.toDouble - 1) * sumOfSq
          (lambda, mean, sd)
        }.toArray
        Some(lambda_with_mse_sd)
      }else{
        None
      }

    println("Performance: " + lambda_with_mse.toSeq.sortBy(_._2))

    // find min test MSE
    val min_test_mse = lambda_with_mse.minBy(_._2)

    // find corresponding lambda
    val min_lambda = min_test_mse._1

    println("Local CV: Best lambda is " + min_lambda + " with MSE " + min_test_mse._2)

    (min_lambda, lambda_with_mse_stats)
  }

}
