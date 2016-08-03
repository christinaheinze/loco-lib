package LOCO.utils

import breeze.linalg._
import breeze.stats.mean
import breeze.numerics.abs
import scala.collection.mutable

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import preprocessingUtils.FeatureVectorLP
import preprocessing.createLocalMatrices

object LOCOUtils {

  /**
   *  Computes the classification error, when the data is distributed over columns
   */
  def computeClassificationError_overCols(
                      sc : SparkContext,
                      coefficientVector : DenseVector[Double],
                      localMatsRDD: RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))],
                      response : DenseVector[Double]
                      ) = {

    // broadcast coefficients
    val betas = sc.broadcast(coefficientVector)

    // compute predictions
    val predictions : DenseVector[Double] =
      localMatsRDD.map{
        case(workerID, (indicesList, localRawFeatureMatrix, localRawFeatureTestMatrix)) =>
          localPrediction((workerID, (indicesList, localRawFeatureMatrix)), betas.value)
      }.reduce(_ + _)

    val responseTimesPredictions: DenseVector[Double] = response :* predictions
    mean(responseTimesPredictions.map(x => if(x > 0.0) 0.0 else 1.0))
  }

  def computeClassificationErrorLogistic_overCols(sc : SparkContext,
                                                  coefficientVector : DenseVector[Double],
                                                  localMatsRDD: RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))],
                                                  response : DenseVector[Double]
                                                   ) = {

    // broadcast coefficients
    val betas = sc.broadcast(coefficientVector)

    // compute predictions
    val predictions : DenseVector[Double] =
      localMatsRDD.map{
        case(workerID, (indicesList, localRawFeatureMatrix, localRawFeatureTestMatrix)) =>
          localPrediction((workerID, (indicesList, localRawFeatureMatrix)), betas.value)
      }.reduce(_ + _)

    val probabilities = predictions.map(x => 1/(1+math.exp(-x)))
    print("\nProbs:" + probabilities + "\n")
    val mappedToOutcome: DenseVector[Double] = probabilities.map(x => if(x > 0.5) 1.0 else -1.0)
    0.5*sum(abs(mappedToOutcome - response))/predictions.length.toDouble
  }

  /**
   *  Computes the mean squared error, when the data is distributed over columns
   */
  def compute_MSE_overCols(
                      sc : SparkContext,
                      coefficientVector : DenseVector[Double],
                      localMatsRDD: RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))],
                      response : DenseVector[Double]
                      ) = {

    // broadcast coefficients
    val betas = sc.broadcast(coefficientVector)

    // compute predictions
    val predictions : DenseVector[Double] =
      localMatsRDD.map{
        case(workerID, (indicesList, localRawFeatureMatrix, localRawFeatureTestMatrix)) =>
          localPrediction((workerID, (indicesList, localRawFeatureMatrix)), betas.value)
      }.reduce(_ + _)

      math.pow(norm(response - predictions), 2)/predictions.length.toDouble
  }

  /**
   *  Computes the mean squared error, standardized by the squared norm of the centered response,
   *  when the data is distributed over columns
   */
  def compute_standardizedMSE_overCols(
                      sc : SparkContext,
                      coefficientVector : DenseVector[Double],
                      localMatsRDD: RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))],
                      response : DenseVector[Double]) : (Double, Double) = {

    // broadcast coefficients
    val betas = sc.broadcast(coefficientVector)

    // compute predictions
    val predictions : DenseVector[Double] =
      localMatsRDD.map{
        case(workerID, (indicesList, localRawFeatureMatrix, localRawFeatureTestMatrix)) =>
         localPrediction((workerID, (indicesList, localRawFeatureMatrix)), betas.value)
      }.reduce(_ + _)

    // compute MSE
    val num : Double = math.pow(norm(response - predictions), 2)
    val meanResponse = sum(response)/response.length.toDouble
    val denom : Double = math.pow(norm(response - meanResponse), 2)

    (num/denom, num/response.length.toDouble)
  }

  def localPrediction(
                     localRawFeatures : (Int, (List[Int], Matrix[Double])),
                     coefficientVector : DenseVector[Double]
                     ) : DenseVector[Double] = {

    // sort coefficientVector according to list of coefficients and compute prediction
    val coefIndices = localRawFeatures._2._1
    val localPrediction = localRawFeatures._2._2 * coefficientVector(coefIndices)
    localPrediction.toDenseVector
  }


  /** Computes, prints and saves summary statistics. **/
  def printSummaryStatistics(
     sc : SparkContext,
     classification : Boolean,
     logistic : Boolean,
     numIterations : Int,
     timeStampStart : Long,
     timeDifference : Long,
     RPTime : Long,
     communicationTime : Long,
     restTime : Long,
     CVTime : Long,
     betaLoco : DenseVector[Double],
     trainingData : RDD[FeatureVectorLP],
     testData : RDD[FeatureVectorLP],
     responseTrain : DenseVector[Double],
     responseTest : DenseVector[Double],
     trainingDatafile : String,
     testDatafile : String,
     responsePathTrain : String,
     responsePathTest : String,
     nPartitions : Int,
     nExecutors : Int,
     nFeatsProj : Int,
     projection : String,
     useSparseStructure : Boolean,
     concatenate : Boolean,
     lambda : Double,
     CV : Boolean,
     lambdaSeq : Seq[Double],
     kFold : Int,
     seed : Int,
     lambdaGlobal : Double,
     checkDualityGap : Boolean,
     stoppingDualityGap : Double,
     privateLOCO : Boolean,
     privateCV : Boolean,
     privateEps : Double,
     privateDelta : Double,
     saveToHDFS : Boolean,
     directoryName : String,
     localLambdas : Option[DenseVector[Double]],
     privateCVStats : Option[Array[Array[(Double, Double, Double)]]],
     globalCVStats : Option[Array[(Double, Double, Double)]],
     debug : Boolean) : Unit = {


    // print estimates
    println("\nEstimates for coefficients (only print first 20): beta_loco = ")
    val printUntil = math.min(20, betaLoco.length)
    println(Vector(betaLoco(0 until printUntil).toArray))

    val MSE_train =
      if(classification){
        val misClassError =
          if(logistic){
            computeClassificationErrorLogistic_overCols(
              sc,
              betaLoco,
              createLocalMatrices(trainingData, useSparseStructure, responseTrain.length, null),
              responseTrain
            )
          }else{
            computeClassificationError_overCols(
              sc,
              betaLoco,
              createLocalMatrices(trainingData, useSparseStructure, responseTrain.length, null),
              responseTrain
            )
          }

        (misClassError, misClassError)
      }else{
        compute_standardizedMSE_overCols(
          sc,
          betaLoco,
          createLocalMatrices(trainingData, useSparseStructure, responseTrain.length, null),
          responseTrain
          )
      }

    if (classification)
      println("Misclassification error on training set = " + MSE_train._1)
    else{
      println("Training Mean Squared Error = " + MSE_train._1)
      println("Training Mean Squared Error (unstandardized) = " + MSE_train._2)
    }


    val MSE_test =
      if(classification){
        val misClassTest =
        if(logistic){
          computeClassificationErrorLogistic_overCols(
            sc,
            betaLoco,
            createLocalMatrices(testData, useSparseStructure, responseTest.length, null),
            responseTest
          )
        }else{
          computeClassificationError_overCols(
            sc,
            betaLoco,
            createLocalMatrices(testData, useSparseStructure, responseTest.length, null),
            responseTest
          )
        }

        (misClassTest, misClassTest)
      }else{
        compute_standardizedMSE_overCols(
          sc,
          betaLoco,
          preprocessing.createLocalMatrices(testData, useSparseStructure, responseTest.length, null),
          responseTest
        )
      }

    if(classification)
      println("Misclassification error on test set = " + MSE_test._1)
    else{
      println("Test Mean Squared Error = " + MSE_test._1)
      println("Test Mean Squared Error (unstandardized) = " + MSE_test._2)
    }


    // save test MSE to file
    if(saveToHDFS)
      sc.parallelize(List(MSE_test._1.toString()), 1)
        .saveAsTextFile(directoryName + "/test_MSE_" + timeStampStart)
    else
      scala.tools.nsc.io.File(directoryName + "/test_MSE_" + timeStampStart +  ".txt")
        .writeAll(MSE_test._1.toString)

    // save beta to file
    if(saveToHDFS)
      sc.parallelize(List(betaLoco.toString()), 1)
        .saveAsTextFile(directoryName + "/beta_" + timeStampStart)
    else
      scala.tools.nsc.io.File(directoryName + "/beta_" + timeStampStart + ".txt")
        .writeAll(betaLoco.toString)

    // print and save running time
    println("LOCO took " + timeDifference + " msecs to run.")
    if(saveToHDFS)
      sc.parallelize(List(timeDifference.toString()), 1)
        .saveAsTextFile(directoryName + "/running_time_in_msecs_" + timeStampStart)
    else
      scala.tools.nsc.io.File(directoryName + "/running_time_in_msecs_" + timeStampStart +  ".txt")
        .writeAll(timeDifference.toString)

    // save configurations
    val build = new mutable.StringBuilder()

    build.append("\nTraining MSE :           " + MSE_train._1)
    build.append(if(!classification)
                 "\nTraining MSE (unstd.) :  " + MSE_train._2)
    build.append("\nTest MSE:                " + MSE_test._1)
    build.append(if(!classification)
                 "\nTest MSE  (unstd.) :     " + MSE_test._2)
    build.append("\nclassification:          " + classification)
    build.append("\nprivateLOCO:             " + privateLOCO)
    build.append("\nprivateEps:              " + privateEps)
    build.append("\nprivateDelta:            " + privateDelta)
    build.append("\nnumIterations:           " + numIterations)
    build.append("\ncheckDualityGap:         " + checkDualityGap)
    build.append("\nstoppingDualityGap:      " + stoppingDualityGap)
    build.append("\nnPartitions:             " + nPartitions)
    build.append("\nnExecutors:              " + nExecutors)
    build.append("\ntrainingDatafile:        " + trainingDatafile)
    build.append("\nresponsePathTrain:       " + responsePathTrain)
    build.append("\ntestDatafile:            " + testDatafile)
    build.append("\nresponsePathTest:        " + responsePathTest)
    build.append("\nuseSparseStructure:      " + useSparseStructure)
    build.append("\nseed:                    " + seed)
    build.append("\nProjection:              " + projection)
    build.append("\nnFeatsProj:              " + nFeatsProj)
    build.append("\nconcatenate:             " + concatenate)
    build.append("\nCV:                      " + CV)
    build.append("\nprivateCV                " + privateCV)
    build.append(if(CV)
                 "\nlambdaGlobal:            " + lambdaGlobal)
    build.append("\nRun time LOCO:           " + timeDifference)
    build.append("\nRP time LOCO:            " + RPTime)
    build.append("\nCommunication time LOCO: " + communicationTime)
    build.append("\nRest time LOCO:          " + restTime)
    build.append("\nCV time LOCO:            " + CVTime)
    build.append(if(!CV & !privateCV)
                 "\nlambda (provided):       " + lambda
                 else{
                 "\nkfold:                   " + kFold +
                 "\nlambdaSeq:               " + lambdaSeq})
    build.append(if(privateCV)
                 "\nlocalLambdas:              " + localLambdas.get)

    if(saveToHDFS)
      sc.parallelize(List(build.toString()), 1)
        .saveAsTextFile(directoryName + "/providedOptions_" + timeStampStart)
    else
      scala.tools.nsc.io.File(directoryName + "/providedOptions_" + timeStampStart + ".txt")
        .writeAll(build.toString())

    if(debug){

      if(privateCV){
        val privateStats: Array[Array[(Double, Double, Double)]] = privateCVStats.get
        for(e <- 0 until privateStats.length){
          val workerE: Array[(Double, Double, Double)] = privateStats(e)
          scala.tools.nsc.io.File(directoryName + "/privateCVStats_" + e + "_" + timeStampStart + ".txt")
            .writeAll(DenseVector(workerE).toString)
        }
      }else if(CV){
        val globalStats: Array[(Double, Double, Double)] = globalCVStats.get
          scala.tools.nsc.io.File(directoryName + "/globalCVStats_" + timeStampStart + ".txt")
            .writeAll(DenseVector(globalStats).toString)
      }

    }

  }
}
