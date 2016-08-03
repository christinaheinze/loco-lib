package LOCO.solvers

import LOCO.utils.CVUtils
import breeze.linalg.{Vector, DenseMatrix, DenseVector}
import breeze.linalg._

import LOCO.utils.ProjectionUtils._
import LOCO.solvers.SDCA.{localSDCA, localSDCARidge, localSDCALogistic}

import scala.collection.Seq


object localDual {

  /**
   * Runs the local optimization algorithm with the random feature representation resulting from
   * adding the random features from the other workers.
   * 
   * @param rawAndRandomFeatsWithIndex Tuple with identifier of partition as key. The value
   *                                   contains the indices of the feature vectors in this
   *                                   partition, the raw features for training, the random features,
   *                                   and optionally the raw features for testing.
   * @param RPsAdded Matrix resulting from adding all random projections.
   * @param response Response vector.
   * @param lambda Lambda to use if no local CV should be done.
   * @param nObs Number of observations.
   * @param classification True if classification problem, otherwise regression.
   * @param numIterations Number of iterations
   * @param nFeatsProj Projection dimension (dimensionality of random feature representation)
   * @param checkDualityGap If the optimizer is SDCA, specify whether the duality gap should be
   *                        computer after each iteration. Note that this is expensive as it
   *                        requires a pass over the entire (local) data set. Should only be used
   *                        for tuning purposes.
   * @param stoppingDualityGap If the optimizer is SDCA, specify the size of the duality gap at
   *                           which the optimization should end. If it is not reached after
   *                           numIterations, the optimization ends nonetheless.
   * @return
   */
  def runLocalDualAdd(
      rawAndRandomFeatsWithIndex: (Int, (List[Int], Matrix[Double], DenseMatrix[Double], Option[Matrix[Double]])) ,
      RPsAdded : Matrix[Double],
      response : Vector[Double],
      lambda : Double,
      nObs : Int,
      classification : Boolean,
      logistic : Boolean,
      numIterations : Int,
      nFeatsProj : Int,
      randomSeed : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double,
      naive : Boolean,
      privateCV : Boolean,
      kfold : Int,
      lambdaSeq : Seq[Double]) : (List[Int], Vector[Double], Option[Double], Option[Array[(Double, Double, Double)]]) = {

    println("Adding random features at worker " + rawAndRandomFeatsWithIndex._1 + "...")

    // subtract own random projection from sum of all random projections
    val randomMats =
      if(!naive)
        randomFeaturesSubtractLocal(rawAndRandomFeatsWithIndex._2._3, RPsAdded)
      else
        null

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (rawAndRandomFeatsWithIndex._2._1, rawAndRandomFeatsWithIndex._2._2)

    // run local dual method on raw and random features
    runLocalDual(
      rawFeaturesWithIndices, randomMats, response, lambda, nObs, classification, logistic, numIterations,
      randomSeed, checkDualityGap, stoppingDualityGap, naive, privateCV, kfold, lambdaSeq)
  }


  def runLocalDualConcatenate(
      rawFeatsWithIndex: (Int, (List[Int], Matrix[Double])),
      RPsMap : collection.Map[Int, Matrix[Double]],
      response : Vector[Double],
      lambda : Double,
      nObs : Int,
      classification : Boolean,
      logistic : Boolean,
      numIterations : Int,
      nFeatsProj : Int,
      randomSeed : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double,
      naive : Boolean,
      privateCV : Boolean,
      kfold : Int,
      lambdaSeq : Seq[Double]) : (List[Int], Vector[Double], Option[Double], Option[Array[(Double, Double, Double)]]) = {

    println("Concatenating random features at worker " + rawFeatsWithIndex._1 + "...")

    // concatenate all random projections, except the one from same worker
    val randomMats =
      if(!naive)
        randomFeaturesConcatenateOrAddAtWorker(rawFeatsWithIndex._1, RPsMap, true)
      else
        null

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (rawFeatsWithIndex._2._1, rawFeatsWithIndex._2._2)

    // run local dual method on raw and random features
    runLocalDual(
      rawFeaturesWithIndices, randomMats, response, lambda, nObs, classification, logistic, numIterations,
      randomSeed, checkDualityGap, stoppingDualityGap, naive, privateCV, kfold, lambdaSeq)

  }


  def runLocalDual(
      matrixWithIndex: (List[Int], Matrix[Double]),
      randomMats : Matrix[Double],
      response : Vector[Double],
      lambdaProvided : Double,
      nObs : Int,
      classification : Boolean,
      logistic : Boolean,
      numIterations : Int,
      randomSeed : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double,
      naive : Boolean,
      privateCV : Boolean,
      kfold : Int,
      lambdaSeq : Seq[Double]) : (List[Int], Vector[Double], Option[Double], Option[Array[(Double, Double, Double)]]) = {

    // cast to dense matrix again
    val rawFeatures = matrixWithIndex._2.toDenseMatrix

    // create design matrix by concatenating raw and random features
    val designMatUnscaled = if(!naive) DenseMatrix.horzcat(rawFeatures, randomMats.toDenseMatrix) else rawFeatures

    val designMat = breeze.linalg.scale(designMatUnscaled, center = false, scale = false)

    // total number of features in local design matrix
    val numFeatures = designMat.cols
    val numRawFeatures = rawFeatures.cols

    // find best lambda if local CV has been chosen, otherwise use provided lambda
    val (lambda, stats) : (Double, Option[Array[(Double, Double, Double)]])  =
      if(privateCV){
        println("Running local CV... ")
        CVUtils.crossValidationDualLocal(
          designMat, response, lambdaSeq, kfold, randomSeed, nObs, numIterations, numFeatures,
          classification, logistic, checkDualityGap, stoppingDualityGap, numRawFeatures)
      }else{
        (lambdaProvided, null)
      }


    // train on full training set with min_lambda to find dual variables alpha
    val alpha: Vector[Double] =
        if(classification) {
          if(logistic){
            localSDCALogistic(
              designMat, response, numIterations, lambda, nObs, numFeatures, randomSeed,
              checkDualityGap, stoppingDualityGap)
          }else{
            localSDCA(
              designMat, response, numIterations, lambda, nObs, numFeatures, randomSeed,
              checkDualityGap, stoppingDualityGap)
          }

        }else{
          localSDCARidge(
            designMat, response, numIterations, lambda, nObs, numFeatures, randomSeed,
            checkDualityGap, stoppingDualityGap)
        }
    
    // map dual to primal variables and scale correctly
    val primalVariables : DenseMatrix[Double] = rawFeatures.t * new DenseMatrix(nObs, 1, alpha.toArray)
    val scaling = 1.0/(nObs*lambda)
    val beta_hat = primalVariables.toDenseVector * scaling

    val localLambda =
      if(privateCV){
        Some(lambda)
      }else{
        None
      }

    // return column indices and coefficient vector
    (matrixWithIndex._1, beta_hat, localLambda, stats)
  }



  def runLocalDualAdd_lambdaSeq(
      matrixWithIndex: (Int, (List[Int], Matrix[Double], Matrix[Double], Option[Matrix[Double]])),
      RPsAdded : Matrix[Double],
      response : Vector[Double],
      lambdaSeq : Seq[Double],
      nObs : Int,
      classification : Boolean,
      logistic : Boolean,
      numIterations : Int,
      nFeatsProj : Int,
      seed : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double,
      naive : Boolean) : Array[(Double, DenseVector[Double])]  = {

    // subtract own random projection from sum of all random projections
    val randomMats = randomFeaturesSubtractLocal(matrixWithIndex._2._3, RPsAdded)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (matrixWithIndex._2._1, matrixWithIndex._2._2, matrixWithIndex._2._4)

    // run local dual method on raw and random features for lambda sequence
    runLocalDual_lambdaSeq(
      rawFeaturesWithIndices, randomMats, response, lambdaSeq, nObs, classification, logistic, numIterations,
      seed, checkDualityGap, stoppingDualityGap, naive)
  }


  def runLocalDualConcatenate_lambdaSeq(
      matrixWithIndex: (Int, (List[Int], Matrix[Double], Option[Matrix[Double]])),
      RPsMap : collection.Map[Int, Matrix[Double]],
      response : Vector[Double],
      lambdaSeq : Seq[Double],
      nObs : Int,
      classification : Boolean,
      logistic : Boolean,
      numIterations : Int,
      nFeatsProj : Int,
      seed : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double,
      naive : Boolean) : Array[(Double, DenseVector[Double])]  = {

    // concatenate all random projections, except the one from same worker
    val randomMats =
      if(!naive)
        randomFeaturesConcatenateOrAddAtWorker(matrixWithIndex._1, RPsMap, true)
      else
        null

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (matrixWithIndex._2._1, matrixWithIndex._2._2, matrixWithIndex._2._3)

    // run local dual method on raw and random features for lambda sequence
    runLocalDual_lambdaSeq(
      rawFeaturesWithIndices, randomMats, response, lambdaSeq, nObs, classification, logistic, numIterations,
      seed, checkDualityGap, stoppingDualityGap, naive)
  }


  def runLocalDual_lambdaSeq(
      matrixWithIndex: (List[Int], Matrix[Double], Option[Matrix[Double]]),
      randomMats: Matrix[Double],
      response : Vector[Double],
      lambdaSeq : Seq[Double],
      nObs : Int,
      classification : Boolean,
      logistic : Boolean,
      numIterations : Int,
      randomSeed : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double,
      naive : Boolean) : Array[(Double, DenseVector[Double])] = {

    // cast to dense matrix
    val rawFeatures = matrixWithIndex._2.toDenseMatrix

    // create design matrix by concatenating raw and random features
    val designMat = if(!naive) DenseMatrix.horzcat(rawFeatures, randomMats) else rawFeatures

    val numFeatures = designMat.cols

    lambdaSeq.map{currentLambda =>
      // train
      val alpha =
          if(classification){
            if(logistic){
              localSDCALogistic(
                designMat, response, numIterations, currentLambda, nObs, numFeatures,
                randomSeed, checkDualityGap, stoppingDualityGap)
            }else{
              localSDCA(
                designMat, response, numIterations, currentLambda, nObs, numFeatures,
                randomSeed, checkDualityGap, stoppingDualityGap)
            }

          }else{
            localSDCARidge(
              designMat, response, numIterations, currentLambda, nObs, numFeatures,
              randomSeed, checkDualityGap, stoppingDualityGap)
          }

      val primalVarsNotScaled : DenseMatrix[Double] = rawFeatures.t * new DenseMatrix(nObs, 1, alpha.toArray)
      val scaling = 1.0/(nObs*currentLambda)
      val beta_hat: DenseVector[Double] = primalVarsNotScaled.toDenseVector * scaling

      // get local test observations
      val rawFeaturesTest : DenseMatrix[Double] = matrixWithIndex._3.get.toDenseMatrix

      // compute local prediction
      val localPrediction : DenseVector[Double] = rawFeaturesTest * beta_hat

      // return column indices and coefficient vector
      (currentLambda, localPrediction)
    }.toArray

  }

}