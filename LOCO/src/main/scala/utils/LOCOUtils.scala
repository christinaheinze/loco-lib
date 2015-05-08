package LOCO.utils

import breeze.linalg.Vector
import scala.collection.mutable

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import preprocessingUtils.DataPoint
import preprocessingUtils.utils.metrics

object LOCOUtils {

  /**
   * Checks whether the provided projection dimension is indeed smaller than the number
   * of raw features that are to be compressed.
   */
  def isValidProjectionDim(
      nProjectionDim : Int,
      nPartitions : Int,
      nFeatures : Int) : Boolean = {

    // compute number of raw features on one worker
    val nRawFeatures = nFeatures/nPartitions + 1

    // projection dimension needs to be smaller
    nProjectionDim < nRawFeatures
  }

  /** Computes, prints and saves summary statistics. **/
  def printSummaryStatistics(
     sc : SparkContext,
     classification : Boolean,
     optimizer : String,
     numIterations : Int,
     timeStampStart : Long,
     timeDifference : Long,
     betaLoco : Vector[Double],
     trainingDataNotCentered : RDD[DataPoint],
     testDataNotCentered : RDD[DataPoint],
     center : Boolean,
     centerFeaturesOnly : Boolean,
     meanResponse : Double,
     colMeans: Vector[Double],
     dataFormat : String,
     separateTrainTestFiles : Boolean,
     trainingDatafile : String,
     testDatafile : String,
     dataFile : String,
     proportionTest : Double,
     nPartitions : Int,
     nExecutors : Int,
     nFeatsProj : Int,
     projection : String,
     flagFFTW : Int,
     concatenate : Boolean,
     lambda : Double,
     CVKind : String,
     lambdaSeq : Seq[Double],
     kFold : Int,
     seed : Int,
     lambdaGlobal : Double,
     checkDualityGap : Boolean,
     stoppingDualityGap : Double,
     saveToHDFS : Boolean,
     directoryName : String) : Unit = {


    // print estimates
    println("\nEstimates for coefficients (only print first 20): beta_loco = ")
    val printUntil = math.min(20, betaLoco.length)
    println(Vector(betaLoco(0 until printUntil).toArray))

    val colMeansBroadcast = if(center || centerFeaturesOnly) sc.broadcast(colMeans) else null

    // center training data by row
    val trainingData_byRow_centered = trainingDataNotCentered.map { elem =>
      if (center) DataPoint(elem.label - meanResponse, elem.features - colMeansBroadcast.value)
      else if (centerFeaturesOnly) DataPoint(elem.label, elem.features - colMeansBroadcast.value)
      else elem
    }.cache()

    // center test data by row
    val testData_byRow_centered = testDataNotCentered.map { elem =>
      if (center) DataPoint(elem.label - meanResponse, elem.features - colMeansBroadcast.value)
      else if (centerFeaturesOnly) DataPoint(elem.label, elem.features - colMeansBroadcast.value)
      else elem
    }.cache()

    val MSE_train =
      if(classification){
        metrics.computeClassificationError(betaLoco, trainingData_byRow_centered)
      }else{
        metrics.compute_standardizedMSE(betaLoco, trainingData_byRow_centered)
      }

    if (classification)
      println("Misclassification error on training set = " + MSE_train)
    else
      println("Training Mean Squared Error = " + MSE_train)

    trainingData_byRow_centered.unpersist()

    val MSE_test =
      if(classification){
        metrics.computeClassificationError(betaLoco, testData_byRow_centered)
      }else{
        metrics.compute_standardizedMSE(betaLoco, testData_byRow_centered)
      }

    if(classification)
      println("Misclassification error on test set = " + MSE_test)
    else
      println("Test Mean Squared Error = " + MSE_test)


    testData_byRow_centered.unpersist()

    // save test MSE to file
    if(saveToHDFS)
      sc.parallelize(List(MSE_test.toString()), 1)
        .saveAsTextFile(directoryName + "/test_MSE_" + timeStampStart)
    else
      scala.tools.nsc.io.File(directoryName + "/test_MSE_" + timeStampStart +  ".txt")
        .writeAll(MSE_test.toString)

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

    build.append("\nTraining MSE :           " + MSE_train)
    build.append("\nTest MSE:                " + MSE_test)
    build.append("\nclassification:          " + classification)
    build.append("\noptimizer:               " + optimizer)
    build.append("\nnumIterations:           " + numIterations)
    build.append("\ncheckDualityGap:         " + checkDualityGap)
    build.append("\nstoppingDualityGap:      " + stoppingDualityGap)
    build.append("\nnPartitions:             " + nPartitions)
    build.append("\nnExecutors:              " + nExecutors)
    build.append("\ndataFormat:              " + dataFormat)
    build.append(if(separateTrainTestFiles){
                 "\ntrainingDatafile:        " + trainingDatafile +
                 "\ntestDatafile:            " + testDatafile}
                 else{
                 "\ndataFile:                " + dataFile +
                 "\nproportionTest:          " + proportionTest})
    build.append("\nseed:                    " + seed)

    build.append("\ncenter:                  " + center)
    build.append("\ncenterFeaturesOnly:      " + centerFeaturesOnly)

    build.append("\nProjection:              " + projection)
    build.append("\nflagFFTW:                " + flagFFTW)
    build.append("\nnFeatsProj:              " + nFeatsProj)
    build.append("\nconcatenate:             " + concatenate)
    build.append("\nCVKind:                  " + CVKind)
    build.append(if(CVKind == "global")
                 "\nlambdaGlobal:            " + lambdaGlobal)
    build.append("\nRun time LOCO:           " + timeDifference)
    build.append(if(CVKind == "none")
                 "\nlambda (provided):       " + lambda
                 else{
                 "\nkfold:                   " + kFold +
                 "\nlambdaSeq:               " + lambdaSeq})

    if(saveToHDFS)
      sc.parallelize(List(build.toString()), 1)
        .saveAsTextFile(directoryName + "/providedOptions_" + timeStampStart)
    else
      scala.tools.nsc.io.File(directoryName + "/providedOptions_" + timeStampStart + ".txt")
        .writeAll(build.toString())

  }


}
