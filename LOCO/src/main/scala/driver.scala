package LOCO


import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.log4j.Logger
import org.apache.log4j.Level

import preprocessingUtils.DataPoint
import preprocessingUtils.loadData.load
import preprocessingUtils.loadData.load._

import LOCO.solvers.runLOCO
import LOCO.utils.LOCOUtils._
import LOCO.utils.CVUtils


object driver {

  def main(args: Array[String]): Unit = {

    // get start timestamp of application
    val globalStartTime = System.currentTimeMillis

    // parse input options
    val options =  args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt) => (opt -> "true")
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap


    // parse and read in inputs

    // 1) input and output options

    // output directory
    val outdir = options.getOrElse("outdir","")
    // specify whether output shall be saved on HDFS
    val saveToHDFS = options.getOrElse("saveToHDFS", "false").toBoolean
    // how many partitions of the data matrix to use
    val nPartitions = options.getOrElse("nPartitions","4").toInt
    // how many executors are used
    val nExecutors = options.getOrElse("nExecutors","4").toInt

    // "text" or "object"
    val dataFormat = options.getOrElse("dataFormat", "text")
    // "libsvm", "spaces" or "comma"
    val textDataFormat = options.getOrElse("textDataFormat", "spaces")
    // input path
    val dataFile = options.getOrElse("dataFile", "../data/climate_train.txt")
    // provide training and test set as separate files?
    val separateTrainTestFiles = options.getOrElse("separateTrainTestFiles", "true").toBoolean
    // training input path
    val trainingDatafile =
      options.getOrElse("trainingDatafile", "../data/climate_pres_scaled_p2p3_12_train.txt")
    // test input path
    val testDatafile =
      options.getOrElse("testDatafile", "../data/climate_pres_scaled_p2p3_12_test.txt")
    // if only one file is provided, proportion used to test set
    val proportionTest = options.getOrElse("proportionTest", "0.2").toDouble
    // random seed
    val myseed = options.getOrElse("seed", "3").toInt

    // 2) specify algorithm, loss function, and optimizer (if applicable)

    // specify whether classification or ridge regression shall be used
    val classification = options.getOrElse("classification", "false").toBoolean
    // use factorie or SDCA
    val optimizer = options.getOrElse("optimizer", "SDCA")
    // number of iterations used in SDCA
    val numIterations = options.getOrElse("numIterations", "20000").toInt
    // set duality gap as convergence criterion
    val stoppingDualityGap = options.getOrElse("stoppingDualityGap", "0.01").toDouble
    // specify whether duality gap as convergence criterion shall be used
    val checkDualityGap = options.getOrElse("checkDualityGap", "false").toBoolean

    // 3) algorithm-specific inputs

    // center features and response
    val center = options.getOrElse("center", "true").toBoolean
    // center features only
    val centerFeaturesOnly = options.getOrElse("centerFeaturesOnly", "false").toBoolean
    // specify projection (sparse or SDCT)
    val projection = options.getOrElse("projection", "sparse")
    // shall sparse data structures be used?
    val useSparseStructure = options.getOrElse("useSparseStructure", "false").toBoolean
    // specify flag for SDCT/FFTW: 64 corresponds to FFTW_ESTIMATE, 0 corresponds to FFTW_MEASURE
    val flagFFTW = options.getOrElse("flagFFTW", "64").toInt
    // specify projection dimension
    val nFeatsProj = options.getOrElse("nFeatsProj", "400").toInt
    // concatenate or add
    val concatenate = options.getOrElse("concatenate", "false").toBoolean
    // cross validation: "global", "local", or "none"
    val CVKind = options.getOrElse("CVKind", "none")
    // k for k-fold CV
    val kfold = options.getOrElse("kfold", "5").toInt
    // regularization parameter sequence start used in CV
    val lambdaSeqFrom = options.getOrElse("lambdaSeqFrom", "65").toDouble
    // regularization parameter sequence end used in CV
    val lambdaSeqTo = options.getOrElse("lambdaSeqTo", "66").toDouble
    // regularization parameter sequence step size used in CV
    val lambdaSeqBy = options.getOrElse("lambdaSeqBy", "1").toDouble
    // create lambda sequence
    val lambdaSeq = lambdaSeqFrom to lambdaSeqTo by lambdaSeqBy
    // regularization parameter to be used if CVKind == "none"
    val lambda = options.getOrElse("lambda", "70").toDouble

    // print out inputs
    println("\nSpecify input and output options: ")
    println("dataFormat:                 " + dataFormat)
    if(dataFormat == "text"){
      println("textDataFormat:             " + textDataFormat)
    }
    println("separateTrainTestFiles:     " + separateTrainTestFiles)
    if(separateTrainTestFiles){
      println("trainingDatafile:           " + trainingDatafile)
      println("testDatafile:               " + testDatafile)
    }else {
      println("dataFile:                   " + dataFile)
      println("proportionTest:             " + proportionTest)
    }
    println("outdir:                     " + outdir)
    println("saveToHDFS:                 " + saveToHDFS)
    println("seed:                       " + myseed)

    println("\nSpecify number of partitions, " +
      "algorithm, loss function, and optimizer (if applicable): ")
    println("nPartitions:                " + nPartitions)
    println("nExecutors:                 " + nExecutors)
    println("classification:             " + classification)
    println("optimizer:                  " + optimizer)
    println("numIterations:              " + numIterations)
    println("checkDualityGap:            " + checkDualityGap)
    println("stoppingDualityGap:         " + stoppingDualityGap)

    println("\nAlgorithm-specific inputs: ")
    println("center:                     " + center)
    println("centerFeaturesOnly:         " + centerFeaturesOnly)
    println("projection:                 " + projection)
    println("useSparseStructure:         " + useSparseStructure)
    println("flagFFTW:                   " + flagFFTW)
    println("nFeatsProj:                 " + nFeatsProj)
    println("concatenate:                " + concatenate)
    println("CVKind:                     " + CVKind)
    println("kfold:                      " + kfold)
    if(CVKind != "none"){
      println("lambdaSeq:                  " + lambdaSeq)
    }else{
      println("lambda:                     " + lambda)
    }

    // create folders for output: top-level folder for all results
    val directoryNameTopLevel: String = outdir + "output"
    val dirOutput: java.io.File = new java.io.File(directoryNameTopLevel)
    if(!dirOutput.exists()){
      dirOutput.mkdir()
    }

    // folder for results of specific run
    // use globalStartTime as identifying timestamp
    val directoryNameResultsFolder = directoryNameTopLevel + "/" + globalStartTime.toString
    val dirResults: java.io.File = new java.io.File(directoryNameResultsFolder)
    if(!dirResults.exists()){
      dirResults.mkdir()
    }

    // start spark context
    val conf = new SparkConf()//.setMaster("local[4]")
      .setAppName("LOCO")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)

    // configuration logging
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    // read in training and test data, distribute over rows
    val (training : RDD[DataPoint], test : RDD[DataPoint]) =
      dataFormat match {

        // input files are text files
        case "text" => {
          val (training_temp, test_temp) =
            load.readTextFiles(
              sc, dataFile, nPartitions, textDataFormat, separateTrainTestFiles,
              trainingDatafile, testDatafile, proportionTest, myseed)

          // convert RDD[Array(Double)] to RDD[DataPoint]
          (training_temp.map(x => doubleArrayToDataPoint(x)),
            test_temp.map(x => doubleArrayToDataPoint(x)))
        }

        // input files are object files
        case "object" =>
          load.readObjectFiles[DataPoint](
            sc, dataFile, nPartitions, separateTrainTestFiles, trainingDatafile,
            testDatafile, proportionTest, myseed)

        // throw exception if another option is given
        case _ => throw new Error("No such data format option (use text or object)!")
      }

    // if cross validation is chosen to be "global", cross-validate
    // targeting the global prediction error
    val lambdaGlobal =
      if(CVKind == "global"){
        CVUtils.globalCV(
          sc, classification, myseed, training,  center, centerFeaturesOnly, nPartitions,
          nExecutors, projection, flagFFTW, useSparseStructure,
          concatenate, nFeatsProj, lambdaSeq, kfold, optimizer,
          numIterations, checkDualityGap, stoppingDualityGap)
      }else{
        lambda
      }

    // compute LOCO coefficients
    val (betaLoco, startTime, afterRPTime, colMeans, meanResponse) =
      runLOCO.run(
        sc, classification, myseed, training, center, centerFeaturesOnly, nPartitions, nExecutors,
        projection, flagFFTW, useSparseStructure,
        concatenate, nFeatsProj, lambdaGlobal, CVKind, lambdaSeq, kfold,
        optimizer, numIterations, checkDualityGap, stoppingDualityGap)

    // get second timestamp needed to time LOCO and compute time difference
    val endTime = System.currentTimeMillis
    val runTime = endTime - startTime
    val RPTime = afterRPTime - startTime
    val restTime = runTime - RPTime

    // print summary stats
    printSummaryStatistics(
      sc, classification, optimizer, numIterations, startTime, runTime, RPTime, restTime,
      betaLoco, training, test, center, centerFeaturesOnly, meanResponse, colMeans, dataFormat,
      separateTrainTestFiles, trainingDatafile, testDatafile, dataFile, proportionTest, nPartitions,
      nExecutors, nFeatsProj, projection, flagFFTW, useSparseStructure,
      concatenate, lambda, CVKind, lambdaSeq, kfold,
      myseed, lambdaGlobal, checkDualityGap, stoppingDualityGap, saveToHDFS, directoryNameResultsFolder)

    // compute end time of application and compute time needed overall
    val globalEndTime = System.currentTimeMillis
    val timeGlobal = globalEndTime - globalStartTime
    println("In total, app took " + timeGlobal + " ms to run.")

    // stop Spark Context
    sc.stop()
  }
}