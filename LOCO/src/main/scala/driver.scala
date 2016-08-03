package LOCO


import breeze.linalg.{unique, min, DenseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{RangePartitioner, SparkConf, SparkContext}

import org.apache.log4j.Logger
import org.apache.log4j.Level

import preprocessingUtils.FeatureVectorLP
import preprocessingUtils.FeatureVectorLP._
import preprocessingUtils.loadData.load

import LOCO.solvers.runLOCO
import LOCO.utils.LOCOUtils._
import LOCO.utils.CVUtils

import scala.io.Source


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
    val outdir = options.getOrElse("outdir","output")
    // specify whether output shall be saved on HDFS
    val saveToHDFS = options.getOrElse("saveToHDFS", "false").toBoolean
    // specify whether input shall be read from HDFS
    val readFromHDFS = options.getOrElse("readFromHDFS", "false").toBoolean
    // how many partitions of the data matrix to use
    val nPartitions = options.getOrElse("nPartitions","4").toInt
    // how many executors are used
    val nExecutors = options.getOrElse("nExecutors","1").toInt
    // training input path
    val trainingDatafile =
      options.getOrElse("trainingDatafile", "../data/dogs_vs_cats/dogs_vs_cats_small_train-colwise/")
    // test input path
    val testDatafile =
        options.getOrElse("testDatafile", "../data/dogs_vs_cats/dogs_vs_cats_small_test-colwise/")
    // response vector - training
    val responsePathTrain =
      options.getOrElse("responsePathTrain", "../data/dogs_vs_cats/dogs_vs_cats_small_train-responseTrain.txt")
    // response vector - test
    val responsePathTest =
      options.getOrElse("responsePathTest", "../data/dogs_vs_cats/dogs_vs_cats_small_test-responseTest.txt")
    // number of features
    val nFeatsPath = options.getOrElse("nFeats", "../data/dogs_vs_cats/dogs_vs_cats_small_train-nFeats.txt")
    // random seed
    val randomSeed = options.getOrElse("seed", "1").toInt
    // shall sparse data structures be used?
    val useSparseStructure = options.getOrElse("useSparseStructure", "false").toBoolean

    // 2) specify algorithm, loss function, and optimizer (if applicable)

    // specify whether classification or ridge regression shall be used
    val classification = options.getOrElse("classification", "true").toBoolean
    val logistic = options.getOrElse("logistic", "false").toBoolean
    // number of iterations used in SDCA
    val numIterations = options.getOrElse("numIterations", "5000").toInt
    // set duality gap as convergence criterion
    val stoppingDualityGap = options.getOrElse("stoppingDualityGap", "0.01").toDouble
    // specify whether duality gap as convergence criterion shall be used
    val checkDualityGap = options.getOrElse("checkDualityGap", "false").toBoolean
    val privateLOCO = options.getOrElse("private", "false").toBoolean
    val privateCV = options.getOrElse("privateCV", "false").toBoolean
    val privateEps = options.getOrElse("privateEps", "10").toDouble
    val privateDelta = options.getOrElse("privateDelta", "0.05").toDouble

    // 3) algorithm-specific inputs

    // specify projection (sparse or SDCT)
    val projection = options.getOrElse("projection", "SDCT")
    // specify projection dimension
    val nFeatsProj = options.getOrElse("nFeatsProj", "200").toInt
    // concatenate or add
    val concatenate = options.getOrElse("concatenate", "false").toBoolean
    // cross validation
    val CV = options.getOrElse("CV", "false").toBoolean
    // k for k-fold CV
    val kfold = options.getOrElse("kfold", "5").toInt
    // regularization parameter sequence start used in CV
    val lambdaSeqFrom = options.getOrElse("lambdaSeqFrom", "1").toDouble
    // regularization parameter sequence end used in CV
    val lambdaSeqTo = options.getOrElse("lambdaSeqTo", "100").toDouble
    // regularization parameter sequence step size used in CV
    val lambdaSeqBy = options.getOrElse("lambdaSeqBy", "1").toDouble
    // create lambda sequence
    val lambdaSeq = lambdaSeqFrom to lambdaSeqTo by lambdaSeqBy
    // regularization parameter to be used if CVKind == "none"
    val lambda = options.getOrElse("lambda", "4.4").toDouble

    val debug = options.getOrElse("debug", "false").toBoolean
    val partitioner = options.getOrElse("partitioner", "random")
    val partitionPath = options.getOrElse("partitionPath", "/Users/heinzec/Data/simulated/partitions31.txt")

    // print out inputs
    println("\nSpecify input and output options: ")

    println("trainingDatafile:           " + trainingDatafile)
    println("responsePathTrain:          " + responsePathTrain)
    println("testDatafile:               " + testDatafile)
    println("responsePathTest:           " + responsePathTest)
    println("nFeatsPath:                 " + nFeatsPath)
    println("useSparseStructure:         " + useSparseStructure)

    println("outdir:                     " + outdir)
    println("saveToHDFS:                 " + saveToHDFS)
    println("seed:                       " + randomSeed)

    println("\nSpecify number of partitions, " +
      "algorithm, loss function, and optimizer (if applicable): ")
    println("nPartitions:                " + nPartitions)
    println("nExecutors:                 " + nExecutors)
    println("classification:             " + classification)
    println("numIterations:              " + numIterations)
    println("checkDualityGap:            " + checkDualityGap)
    println("stoppingDualityGap:         " + stoppingDualityGap)
    println("privateLOCO:                " + privateLOCO)
    println("privateEps:                 " + privateEps)
    println("privateDelta:               " + privateDelta)


    println("\nAlgorithm-specific inputs: ")
    println("projection:                 " + projection)
    println("nFeatsProj:                 " + nFeatsProj)
    println("concatenate:                " + concatenate)
    println("CV:                         " + CV)
    println("privateCV:                  " + privateCV)
    println("kfold:                      " + kfold)
    if(CV || privateCV){
      println("lambdaSeq:                  " + lambdaSeq)
    }else{
      println("lambda:                     " + lambda)
    }

    // create folders for output: top-level folder for all results
    val directoryNameTopLevel: String = outdir
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

    // read in training and test data, distributed over columns
    val (training : RDD[FeatureVectorLP], test : RDD[FeatureVectorLP]) =
       load.readObjectFiles[FeatureVectorLP](
            sc, null, nPartitions, true, trainingDatafile,
            testDatafile, 0.2, randomSeed)


    val trainingPartitioned: RDD[FeatureVectorLP] = partitioner match{
        case "hash" => {
          // create hash partitioner
          val partitionID = load.readPartitionsFile(partitionPath)

          require(unique(DenseVector(partitionID.map(x => x._1.toDouble))).length == nPartitions)

          val trainingTemp = training.map(x => {
//            print("\npID " + partitionID(x.index)._1 + " vID " + partitionID(x.index)._2 + " index " + x.index)
           require(partitionID(x.index)._2 == x.index)
            (partitionID(x.index)._1, x)
          })

          val customPartitioner = new org.apache.spark.HashPartitioner(nPartitions)

          // repartition
          trainingTemp
            .partitionBy(customPartitioner)
            .map(x => x._2)
            .persist(StorageLevel.MEMORY_AND_DISK)
        }
        case "range" => {
          // create range partitioner
          val trainingTemp = training.map(x=> (x.index, x.observations))
          val customPartitioner = new RangePartitioner(nPartitions, trainingTemp)

          // repartition
          trainingTemp
            .partitionBy(customPartitioner)
            .map(x => FeatureVectorLP(x._1, x._2))
            .persist(StorageLevel.MEMORY_AND_DISK)
        }
        case "random" => {
          // repartition
          training
            .repartition(nPartitions)
            .persist(StorageLevel.MEMORY_AND_DISK)
        }
        case _ =>  throw new IllegalArgumentException("Invalid argument for partitioner : " + partitioner)
      }


    // force evaluation to allow for proper timing
    trainingPartitioned.foreach(x => {})

    // read response vectors
    val responseTrain =
      if(readFromHDFS)
        DenseVector(sc.textFile(responsePathTrain).flatMap(line => line.split(" ")).map(x => x.toDouble).collect())
      else
        DenseVector(load.readResponse(responsePathTrain).toArray)

    val responseTest =
      if(readFromHDFS)
        DenseVector(sc.textFile(responsePathTest).flatMap(line => line.split(" ")).map(x => x.toDouble).collect())
      else
        DenseVector(load.readResponse(responsePathTest).toArray)

    // read number of features
    val nFeats =
      if(readFromHDFS)
        sc.textFile(nFeatsPath).flatMap(line => line.split(" ")).map(x => x.toInt).first()
      else
        Source.fromFile(nFeatsPath).getLines().mkString.toInt

    // start timing for cross validation
    val CVStart = System.currentTimeMillis()

    // cross validation
    val (lambdaCV : Double, globalCVStats: Option[Array[(Double, Double, Double)]]) =
      if(CV){
        CVUtils.globalCV(
          sc, classification, logistic, randomSeed, trainingPartitioned, responseTrain, nFeats,
          nPartitions, nExecutors, projection, useSparseStructure,
          concatenate, nFeatsProj, lambdaSeq, kfold,
          numIterations, checkDualityGap, stoppingDualityGap, privateLOCO, privateEps, privateDelta,
          debug)
      }else{
        (lambda, None)
      }

    // stop timing for cross validation
    val CVTime = System.currentTimeMillis() - CVStart

    // compute LOCO coefficients
    val (betaLoco, startTime, afterRPTime, afterCommTime, localLambdas, privateCVStats) =
      runLOCO.run(
        sc, classification, logistic, randomSeed, trainingPartitioned, responseTrain, nFeats,
        nPartitions, nExecutors, projection, useSparseStructure,
        concatenate, nFeatsProj, lambdaCV,
        numIterations, checkDualityGap, stoppingDualityGap,
        privateLOCO, privateEps, privateDelta, privateCV, kfold, lambdaSeq)

    // get second timestamp needed to time LOCO and compute time difference
    val endTime = System.currentTimeMillis
    val runTime = endTime - startTime
    val RPTime = afterRPTime - startTime
    val communicationTime = afterCommTime - afterRPTime
    val restTime = runTime - RPTime - communicationTime

    // print summary stats
    printSummaryStatistics(
      sc, classification, logistic, numIterations, startTime, runTime, RPTime, communicationTime, restTime, CVTime,
      betaLoco, trainingPartitioned, test, responseTrain, responseTest, trainingDatafile, testDatafile,
      responsePathTrain, responsePathTest, nPartitions, nExecutors, nFeatsProj, projection,
      useSparseStructure, concatenate, lambda, CV, lambdaSeq, kfold, randomSeed, lambdaCV,
      checkDualityGap, stoppingDualityGap, privateLOCO, privateCV, privateEps, privateDelta,
      saveToHDFS, directoryNameResultsFolder, localLambdas, privateCVStats, globalCVStats, debug)

    // compute end time of application and compute time needed overall
    val globalEndTime = System.currentTimeMillis
    val timeGlobal = globalEndTime - globalStartTime
    println("In total, app took " + timeGlobal + " ms to run.")

    // stop Spark Context
    sc.stop()
  }
}