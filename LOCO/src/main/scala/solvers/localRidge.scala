package LOCO.solvers

import breeze.linalg._
import collection.mutable

import LOCO.utils.{Input, Example}
import LOCO.utils.ProjectionUtils._


object localRidge {

  def runLocalRidgeRegressionAdd(
      matrixWithIndex: (Int, (List[Int], DenseMatrix[Double], DenseMatrix[Double])),
      RPsAdded : DenseMatrix[Double],
      response : Vector[Double],
      concatenate : Boolean,
      doCV : Boolean,
      k : Int,
      lambdaSeq : Seq[Double],
      lambda : Double,
      nObs : Int,
      nFeatsProj : Int) : (List[Int], Vector[Double]) = {

    // subtract own random projection from sum of all random projections
    val randomMats = randomFeaturesSubtractLocal(matrixWithIndex, RPsAdded)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (matrixWithIndex._2._1, matrixWithIndex._2._2)

    // run local ridge regression on raw and random features
    runLocalRidgeRegression(
      rawFeaturesWithIndices, randomMats, response, concatenate, doCV, k, lambdaSeq, lambda, nObs)
  }


  def runLocalRidgeRegressionConcatenate(
      matrixWithIndex: (Int, (List[Int], DenseMatrix[Double])),
      RPsMap : collection.Map[Int, DenseMatrix[Double]],
      response : Vector[Double],
      concatenate : Boolean,
      doCV : Boolean,
      k : Int,
      lambdaSeq : Seq[Double],
      lambda : Double,
      nObs : Int,
      nFeatsProj : Int) : (List[Int], Vector[Double]) = {

    // concatenate all random projections, except the one from same worker
    val randomMats = randomFeaturesConcatenateOrAddAtWorker(matrixWithIndex, RPsMap, concatenate)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (matrixWithIndex._2._1, matrixWithIndex._2._2)

    // run local ridge regression on raw and random features
    runLocalRidgeRegression(
      rawFeaturesWithIndices, randomMats, response, concatenate, doCV, k, lambdaSeq, lambda, nObs)

  }


  def runLocalRidgeRegression(
      matrixWithIndex:(List[Int], DenseMatrix[Double]),
      randomMats : DenseMatrix[Double],
      response : Vector[Double],
      concatenate : Boolean,
      doCV : Boolean,
      k : Int,
      lambdaSeq : Seq[Double],
      lambda : Double,
      nObs : Int) : (List[Int], Vector[Double]) = {

    // get number of raw features
    val size_raw = matrixWithIndex._1.length

    // create design matrix by concatenating raw and random features
    val designMat = DenseMatrix.horzcat(matrixWithIndex._2, randomMats)

    // run cross validation

    // set seed
    val myseed = 3

    // find best lambda or use provided lambda
    val min_lambda =
      if(doCV) {
        crossValidationRidgeLocal(designMat, response, lambdaSeq, k, myseed, nObs)
      }else{
        lambda
      }

    // train on full training set with min_lambda
    var myExamples = mutable.ArrayBuffer[Example]()
    
    for (i <- 0 until designMat.rows){
      myExamples += new Example(new Input(designMat(i ,::).t.toArray), response(i))
    }
    
    val regressor = 
      cc.factorie.app.regress.LinearRegressionTrainer
        .train[Input, Example](myExamples, {f => f.input}, min_lambda)
    
    val beta_hat = Vector(regressor.weights.toArray)

    // return column indices and coefficient vector (choose relevant coefficients)
    (matrixWithIndex._1, beta_hat(0 until size_raw))
  }


  def crossValidationRidgeLocal(
      designMat : DenseMatrix[Double], 
      response : Vector[Double], 
      lambdaSeq : Seq[Double], 
      k : Int, 
      seed : Int,
      nObs : Int) : Double = {

    // set seed
    util.Random.setSeed(seed)
    val times = nObs/k + 1

    // create indices for training and test examples
    val shuffledIndices = util.Random.shuffle(List.fill(times)(1 to k).flatten).take(nObs)

    // create training and test sets
    val trainingAndTestSets = (1 to k).map { fold =>

      var myTrainingExamples = mutable.ArrayBuffer[Example]()
      var myTestExamples = mutable.ArrayBuffer[Example]()

      for(ind <- 0 until nObs){
        val bucket = shuffledIndices(ind)
        if(bucket == fold){
          myTestExamples += new Example(new Input(designMat(ind ,::).t.toArray), response(ind))
        }else{
          myTrainingExamples += new Example(new Input(designMat(ind ,::).t.toArray), response(ind))
        }
      }

      (myTrainingExamples, myTestExamples)
    }.toArray

    // for each lambda in sequence, return average MSE
    val k_mse = trainingAndTestSets.flatMap {
      // for each train and test set pair...
      case (train, test) =>
        // for each lambda...
        lambdaSeq.map{ currentLambda =>
          // ... train on train and test on test
          val regressor_k =
            cc.factorie.app.regress.LinearRegressionTrainer
              .train[Input, Example](train, {f => f.input}, currentLambda)

          // compute squared test error for each point
          val test_errors = test.map { testpoint =>
            math.pow(
              (Vector(testpoint.input.value.toArray) dot Vector(regressor_k.weights.toArray))
                - testpoint.label, 2)
          }

          // compute and return the MSE for this lambda and this train/test pair
          (currentLambda, sum(test_errors)/test_errors.length.toDouble)
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


  def runLocalRidgeRegressionAdd_lambdaSeq(
      matrixWithIndex: (Int, (List[Int], DenseMatrix[Double], DenseMatrix[Double])),
      RPsAdded : DenseMatrix[Double],
      response : Vector[Double],
      concatenate : Boolean,
      lambdaSeq : Seq[Double],
      nObs : Int,
      nFeatsProj : Int) : Seq[(Double, (List[Int], Vector[Double]))] = {

    // subtract own random projection from sum of all random projections
    val randomMats = randomFeaturesSubtractLocal(matrixWithIndex, RPsAdded)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (matrixWithIndex._2._1, matrixWithIndex._2._2)

    // run local ridge regression on raw and random features and a sequence of lambda values
    runLocalRidgeRegression_lambdaSeq(
      rawFeaturesWithIndices, randomMats, response, concatenate, lambdaSeq, nObs)
  }


  def runLocalRidgeRegressionConcatenate_lambdaSeq(
      matrixWithIndex: (Int, (List[Int], DenseMatrix[Double])),
      RPsMap : collection.Map[Int, DenseMatrix[Double]],
      response : Vector[Double],
      concatenate : Boolean,
      lambdaSeq : Seq[Double],
      nObs : Int,
      nFeatsProj : Int) : Seq[(Double, (List[Int], Vector[Double]))] = {

    // concatenate all random projections, except the one from same worker
    val randomMats = randomFeaturesConcatenateOrAddAtWorker(matrixWithIndex, RPsMap, concatenate)

    // extract indices of raw feature vectors and corresponding feature vectors
    val rawFeaturesWithIndices = (matrixWithIndex._2._1, matrixWithIndex._2._2)

    // run local ridge regression on raw and random features and a sequence of lambda values

    runLocalRidgeRegression_lambdaSeq(
      rawFeaturesWithIndices, randomMats, response, concatenate, lambdaSeq, nObs)
  }


  def runLocalRidgeRegression_lambdaSeq(
      matrixWithIndex: (List[Int], DenseMatrix[Double]),
      randomMats :  DenseMatrix[Double],
      response : Vector[Double],
      concatenate : Boolean,
      lambdaSeq : Seq[Double],
      nObs : Int
      ) : Seq[(Double, (List[Int], Vector[Double]))] = {


    // get number of raw features
    val size_raw = matrixWithIndex._1.length

    // create design matrix by concatenating raw and random features
    val designMat = DenseMatrix.horzcat(matrixWithIndex._2, randomMats)


    // create training set
    var myExamples = mutable.ArrayBuffer[Example]()
    for (i <- 0 until designMat.rows) {
      myExamples += new Example(new Input(designMat(i ,::).t.toArray), response(i))
    }

    lambdaSeq.map{currentLambda =>
      val regressor =
        cc.factorie.app.regress.LinearRegressionTrainer
          .train[Input, Example](myExamples, {f => f.input}, currentLambda)
      val beta_hat = Vector(regressor.weights.toArray)

      // return column indices and coefficient vector (choose relevant coefficients)
      (currentLambda, (matrixWithIndex._1, beta_hat(0 until size_raw)))
    }
  }


}

