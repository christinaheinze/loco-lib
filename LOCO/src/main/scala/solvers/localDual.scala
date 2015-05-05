package LOCO.solvers

import breeze.linalg.{Vector, DenseMatrix, DenseVector}
import breeze.linalg._

import LOCO.utils.ProjectionUtils._
import LOCO.solvers.SDCA.{localSDCA, localSDCARidge}


object localDual {

  /**
   * Runs the local optimization algorithm with the random feature representation resulting from
   * adding the random features from the other workers.
   * 
   * @param rawAndRandomFeatsWithIndex Tuple with identifier of partition as key. The value
   *                                   contains the indices of the feature vectors in this
   *                                   partition, the raw features and the random features.
   * @param RPsAdded Matrix resulting from adding all random projections.
   * @param response Response vector.
   * @param doCV Set to true if the regularization parameter should be cross-validated locally.
   * @param kFold Number of splits in cross validation.
   * @param lambdaSeq Lambda sequence to use in cross validation.
   * @param lambda Lambda to use if no local CV should be done.
   * @param nObs Number of observations.
   * @param classification True if classification problem, otherwise regression.
   * @param optimizer "SDCA" or "factorie"
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
      rawAndRandomFeatsWithIndex: (Int, (List[Int], DenseMatrix[Double], DenseMatrix[Double])),
      RPsAdded : DenseMatrix[Double],
      response : Vector[Double],
      doCV : Boolean,
      kFold : Int,
      lambdaSeq : Seq[Double],
      lambda : Double,
      nObs : Int,
      classification : Boolean,
      optimizer : String,
      numIterations : Int,
      nFeatsProj : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : (List[Int], Vector[Double]) = {

    // subtract own random projection from sum of all random projections
    val randomMats = randomFeaturesSubtractLocal(rawAndRandomFeatsWithIndex, RPsAdded)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (rawAndRandomFeatsWithIndex._2._1, rawAndRandomFeatsWithIndex._2._2)

    // run local dual method on raw and random features
    runLocalDual(
      rawFeaturesWithIndices, randomMats, response, doCV, kFold, lambdaSeq,
      lambda, nObs, classification, optimizer, numIterations,checkDualityGap, stoppingDualityGap)
  }


  def runLocalDualConcatenate(
      rawFeatsWithIndex: (Int, (List[Int], DenseMatrix[Double])),
      RPsMap : collection.Map[Int, DenseMatrix[Double]],
      response : Vector[Double],
      doCV : Boolean,
      k : Int,
      lambdaSeq : Seq[Double],
      lambda : Double,
      nObs : Int,
      classification : Boolean,
      optimizer : String,
      numIterations : Int,
      nFeatsProj : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : (List[Int], Vector[Double]) = {

    // concatenate all random projections, except the one from same worker
    val randomMats = randomFeaturesConcatenateOrAddAtWorker(rawFeatsWithIndex, RPsMap, true)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (rawFeatsWithIndex._2._1, rawFeatsWithIndex._2._2)

    // run local dual method on raw and random features
    runLocalDual(
      rawFeaturesWithIndices, randomMats, response, doCV, k, lambdaSeq,
      lambda, nObs, classification, optimizer, numIterations, checkDualityGap, stoppingDualityGap)

  }


  def runLocalDual(
      matrixWithIndex: (List[Int], DenseMatrix[Double]),
      randomMats : DenseMatrix[Double],
      response : Vector[Double],
      doCV : Boolean,
      k : Int,
      lambdaSeq : Seq[Double],
      lambda : Double,
      nObs : Int,
      classification : Boolean,
      optimizer : String,
      numIterations : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : (List[Int], Vector[Double]) = {

    // create design matrix by concatenating raw and random features
    val designMat = DenseMatrix.horzcat(matrixWithIndex._2, randomMats)

    // total number of features in local design matrix
    val numFeatures = designMat.cols

    // run cross validation

    // set seed
    val myseed = 3

    // find best lambda if local CV has been chosen, otherwise use provided lambda
    val min_lambda =
      if(doCV){
        crossValidationDualLocal(
          designMat, response, lambdaSeq, k, myseed, nObs, numIterations, numFeatures,
          classification, optimizer, checkDualityGap, stoppingDualityGap)
      }else{
        lambda
      }

    // train on full training set with min_lambda to find dual variables alpha
    val alpha = optimizer match {
      // choose correct combination of solver and problem
      case "SDCA" =>
        if(classification) {
          localSDCA(
            designMat, response, numIterations, min_lambda, nObs, numFeatures, myseed,
            checkDualityGap, stoppingDualityGap)
        }else{
          localSDCARidge(
            designMat, response, numIterations, min_lambda, nObs, numFeatures, myseed,
            checkDualityGap, stoppingDualityGap)
        }

      case "factorie" =>
        new LinearL2SVM(lossType = 1, cost = 1/min_lambda, bias = 0, maxIterations = numIterations)
          .train(designMat, response, 1)
      case _ =>
        throw new IllegalArgumentException("Invalid argument for optimizerClassification : "
          + optimizer)
    }

    // map dual to primal variables and scale correctly
    val primalVariables : DenseMatrix[Double] =
      matrixWithIndex._2.t * new DenseMatrix(nObs, 1, alpha.toArray)
    val scaling = 1.0/(nObs*min_lambda)
    val beta_hat = primalVariables.toDenseVector * scaling

    // return column indices and coefficient vector
    (matrixWithIndex._1, beta_hat)
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
      optimizer : String,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : Double = {

    // set seed
    util.Random.setSeed(myseed)
    val times = nObs/k + 1

    // create indices for training and test examples
    val shuffledIndices = util.Random.shuffle(List.fill(times)(1 to k).flatten).take(nObs)

    // create training and test sets
    val trainingAndTestSets = (1 to k).map { fold =>

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
          val alpha = optimizer match {
            case "SDCA" =>
              if(classification){
                SDCA.localSDCA(
                  train._1, train._2, numIterations, currentLambda, n, numFeatures, myseed,
                  checkDualityGap, stoppingDualityGap)
              }else{
                 SDCA.localSDCARidge(
                   train._1, train._2, numIterations, currentLambda, n, numFeatures, myseed,
                   checkDualityGap, stoppingDualityGap)
              }

            case "factorie" =>
              new LinearL2SVM(lossType = 1, cost = 1/currentLambda, bias = 0,
                maxIterations = numIterations)
                .train(train._1, train._2, 1)
            case _ =>
              throw new IllegalArgumentException("Invalid argument for optimizerClassification : "
                + optimizer)
          }

          val primalVarsNotScaled : DenseMatrix[Double] =
            train._1.t * new DenseMatrix(n, 1, alpha.toArray)
          val scaling = 1.0/(n*currentLambda)
          val beta_hat_k : DenseVector[Double] = primalVarsNotScaled.toDenseVector * scaling

          // compute performance
          val performance =
            if(classification){

              // compute predictions
              val predictions : DenseVector[Double] =
                test._2 :*
                  (test._1 * new DenseMatrix(numFeatures, 1, beta_hat_k.toArray)).toDenseVector

              // compute misclassification error
              predictions
                .toArray
                .map(x => if(x > 0.0) 0.0 else 1.0)
                .sum/test._2.length.toDouble
            }
            else{
              // compute predictions
              val predictions : DenseVector[Double] =
                (test._1 * new DenseMatrix(numFeatures, 1, beta_hat_k.toArray)).toDenseVector

              // compute MSE
              (predictions - test._2).toArray.map(x => x*x).sum/test._2.length.toDouble
            }

          //return lambda and the performance for this lambda and this train/test pair
          (currentLambda, performance)
        }
    }

    // average performance of each lambda over k train/test pairs
    val lambda_with_mse =
      k_mse
        .groupBy(_._1)
        .mapValues(vals => vals.map(elem => elem._2))
        .mapValues(elem => sum(elem)/elem.length.toDouble)

    // find min test MSE
    val min_test_mse = lambda_with_mse.minBy(_._2)

    // find corresponding lambda
    val min_lambda = min_test_mse._1

    println("Local CV: Best lambda is " + min_lambda + " with MSE " + min_test_mse._2)

    min_lambda
  }



  def runLocalDualAdd_lambdaSeq(
      matrixWithIndex: (Int, (List[Int], DenseMatrix[Double], DenseMatrix[Double])),
      RPsAdded : DenseMatrix[Double],
      response : Vector[Double],
      lambdaSeq : Seq[Double],
      nObs : Int,
      seed : Int,
      classification : Boolean,
      optimizer : String,
      numIterations : Int,
      nFeatsProj : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : Seq[(Double, (List[Int], Vector[Double]))] = {

    // subtract own random projection from sum of all random projections
    val randomMats = randomFeaturesSubtractLocal(matrixWithIndex, RPsAdded)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (matrixWithIndex._2._1, matrixWithIndex._2._2)

    // run local dual method on raw and random features for lambda sequence
    runLocalDual_lambdaSeq(
      rawFeaturesWithIndices, randomMats, response, lambdaSeq, nObs, seed,
      classification, optimizer, numIterations, checkDualityGap, stoppingDualityGap)
  }


  def runLocalDualConcatenate_lambdaSeq(
      matrixWithIndex: (Int, (List[Int], DenseMatrix[Double])),
      RPsMap : collection.Map[Int, DenseMatrix[Double]],
      response : Vector[Double],
      lambdaSeq : Seq[Double],
      nObs : Int,
      seed : Int,
      classification : Boolean,
      optimizer : String,
      numIterations : Int,
      nFeatsProj : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : Seq[(Double, (List[Int], Vector[Double]))] = {

    // concatenate all random projections, except the one from same worker
    val randomMats = randomFeaturesConcatenateOrAddAtWorker(matrixWithIndex, RPsMap, true)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (matrixWithIndex._2._1, matrixWithIndex._2._2)

    // run local dual method on raw and random features for lambda sequence
    runLocalDual_lambdaSeq(
      rawFeaturesWithIndices, randomMats, response, lambdaSeq, nObs,  seed,
      classification, optimizer, numIterations, checkDualityGap, stoppingDualityGap)
  }


  def runLocalDual_lambdaSeq(
      matrixWithIndex: (List[Int], DenseMatrix[Double]),
      randomMats: DenseMatrix[Double],
      response : Vector[Double],
      lambdaSeq : Seq[Double],
      nObs : Int,
      myseed : Int,
      classification : Boolean,
      optimizer : String,
      numIterations : Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double) : Seq[(Double, (List[Int], Vector[Double]))] = {

    // create design matrix by concatenating raw and random features
    val designMat = DenseMatrix.horzcat(matrixWithIndex._2, randomMats)

    val numFeatures = designMat.cols

    lambdaSeq.map{currentLambda =>
      // train
      val alpha = optimizer match {
        case "SDCA" =>
          if(classification){
            localSDCA(
              designMat, response, numIterations, currentLambda, nObs, numFeatures,
              myseed, checkDualityGap, stoppingDualityGap)
          }else{
            localSDCARidge(
              designMat, response, numIterations, currentLambda, nObs, numFeatures,
              myseed, checkDualityGap, stoppingDualityGap)
          }

        case "factorie" =>
          new LinearL2SVM(lossType = 1, cost = 1/currentLambda,
            bias = 0, maxIterations = numIterations)
            .train(designMat, response, 1)
        case _ =>
          throw new IllegalArgumentException("Invalid argument for optimizerClassification : "
            + optimizer)
      }

      val primalVarsNotScaled : DenseMatrix[Double] =
        matrixWithIndex._2.t * new DenseMatrix(nObs, 1, alpha.toArray)
      val scaling = 1.0/(nObs*currentLambda)
      val beta_hat = primalVarsNotScaled.toDenseVector * scaling

      // return column indices and coefficient vector
      (currentLambda, (matrixWithIndex._1, beta_hat))
    }
  }

}