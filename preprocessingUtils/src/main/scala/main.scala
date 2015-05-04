package preprocessingUtils

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}

import loadData.load._
import saveData.save


object main {

  def main(args: Array[String]): Unit = {

    // parse input options
    val options = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt) => (opt -> "true")
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap


    // parse and read in inputs

    // output directory
    val outdir = options.getOrElse("outdir","")
    // how many partitions of the data matrix to use
    val nPartitions = options.getOrElse("nPartitions","4").toInt
    // "text" or "object"
    val dataFormat = options.getOrElse("dataFormat", "text")
    // "libsvm", "spaces" or "comma"
    val textDataFormat = options.getOrElse("textDataFormat", "spaces")
    // input path
    val dataFile = options.getOrElse("dataFile", "../data/dogs_vs_cats_n5000.txt")
    // provide training and test set as separate files?
    val separateTrainTestFiles = options.getOrElse("separateTrainTestFiles", "true").toBoolean
    // training input path
    val trainingDatafile =
      options.getOrElse("trainingDatafile", "../data/climate_train.txt")
    // test input path
    val testDatafile =
      options.getOrElse("testDatafile", "../data/climate_test.txt")
    // if only one file is provided, proportion used to test set
    val proportionTest = options.getOrElse("proportionTest", "0.2").toDouble
    // random seed
    val myseed = options.getOrElse("seed", "3").toInt
    // get timestamp as identifier of output files
    val timestamp = System.currentTimeMillis.toString
    // file name for training file output
    val outputTrainFileName =
      options.getOrElse("outputTrainFileName", "output/outTrain" + timestamp)
    // file name for test file output
    val outputTestFileName = options.getOrElse("outputTestFileName", "output/outTest" + timestamp)
    // specify class of output: DataPoint, LabeledPoint or DoubleArray
    val outputClass = options.getOrElse("outputClass", "DataPoint")
    // if two different output formats are desired, set to true
    val twoOutputClasses = options.getOrElse("twoOutputClasses", "true").toBoolean
    // specify second output format
    val secondOutputClass = options.getOrElse("secondOutputClass", "LabeledPoint")
    // center the features to have mean zero
    val centerFeatures = options.getOrElse("centerFeatures", "true").toBoolean
    // center the response to have mean zero
    val centerResponse = options.getOrElse("centerResponse", "true").toBoolean
    // scale the features to have unit variance
    val scaleFeatures = options.getOrElse("scaleFeatures", "true").toBoolean
    // scale the response to have unit variance
    val scaleResponse = options.getOrElse("scaleResponse", "false").toBoolean

    // print out inputs
    println("\nSpecify input and output options: ")
    println("dataFormat:              " + dataFormat)
    if(dataFormat == "text"){
      println("textDataFormat:          " + textDataFormat)
    }
    println("separateTrainTestFiles:  " + separateTrainTestFiles)
    if(separateTrainTestFiles){
      println("trainingDatafile:        " + trainingDatafile)
      println("testDatafile:            " + testDatafile)
    }else {
      println("dataFile:                " + dataFile)
      println("proportionTest:          " + proportionTest)
    }
    println("outdir:                  " + outdir)
    println("seed:                    " + myseed)

    println("nPartitions:             " + nPartitions)
    println("centerResponse:          " + centerResponse)
    println("centerFeatures:          " + centerFeatures)
    println("scaleResponse:           " + scaleResponse)
    println("scaleFeatures:           " + scaleFeatures)

    // create folder for output
    val DirectoryName: String = outdir + "output"
    val dir: java.io.File = new java.io.File(DirectoryName)
    if(!dir.exists()){
      dir.mkdir()
    }

    // start spark context
    val conf = new SparkConf()//.setMaster("local[4]")
      .setAppName("preprocessingUtils")
      .setJars(SparkContext.jarOfObject(this).toSeq)
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)

    // configuration logging
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    // read in training and test data, distribute over rows, and map elements
    // to LabeledPoint so that we can use MLlib's centering and scaling
    val (trainingDataNotCenteredLabeledPoint, testDataNotCenteredLabeledPoint) =
      dataFormat match {
        case "object" =>
          readAndCastObjectFiles(
            sc, dataFile, nPartitions, separateTrainTestFiles, trainingDatafile, testDatafile,
            proportionTest, myseed)
        case "text" =>
          val (train, test) =
            readTextFiles(
              sc, dataFile, nPartitions, textDataFormat, separateTrainTestFiles,
              trainingDatafile, testDatafile, proportionTest, myseed)
          (train.map(x => doubleArrayToLabeledPoint(x)),test.map(x => doubleArrayToLabeledPoint(x)))
        case _ => throw new Error("dataFormat must be \'object\' or \'text\'!")
      }

    // center and/or scale training data
    val trainingDataCentered =
      preprocess
        .transform
        .centerAndScale(
          trainingDataNotCenteredLabeledPoint, centerFeatures, centerResponse,
          scaleFeatures, scaleResponse)

    // center and/or scale test data
    val testDataCentered =
      preprocess
        .transform
        .centerAndScale(
          testDataNotCenteredLabeledPoint, centerFeatures, centerResponse,
          scaleFeatures, scaleResponse)

    // cast in desired format as specified in "outputClass" and save
    save.castAndSave(
      trainingDataCentered, testDataCentered, outputClass, outputTrainFileName, outputTestFileName)

    // if two different output formats are desired, cast in desired format as specified
    // in "secondOutputClass" and save
    if(twoOutputClasses){
      save.castAndSave(
        trainingDataCentered, testDataCentered, secondOutputClass, outputTrainFileName,
        outputTestFileName)
    }

    println("Application finished running successfully!")
  }

}
