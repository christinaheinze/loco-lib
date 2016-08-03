package LOCO.utils

import org.apache.spark.mllib.linalg

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import breeze.linalg._

import org.apache.spark.rdd.RDD

import preprocessingUtils.FeatureVectorLP

object preprocessing {


  /**
   *  Aggregate feature vectors per partition to matrix
   *
   * @param partitionIndex ID for partition
   * @param iter Iterator over elements in partitions. The elements are of type FeatureVector.
   * @param nObs Number of observation in data set.
   * @param trainingAndTestIndices  Tuple containing indices of training and test points
   *
   * @return Iterator with key-value pairs where the key is the partitiion ID and the value
   *         consists of a tuple containing the column indices and the local raw features of
   *         the partition. Optionally, the local raw features are split according to indices in
   *         trainingAndTestIndices. In this case, a second raw feature matrix for testing is
   *         returned.
   */
  def createLocalMatricesDense(
      partitionIndex : Int,
      iter : Iterator[FeatureVectorLP],
      nObs : Int,
      trainingAndTestIndices : (List[Int], List[Int]))
  : Iterator[(Int, (List[Int], DenseMatrix[Double], Option[DenseMatrix[Double]]))] = {

    var columnIndices = new mutable.MutableList[Int]()
    var ArrayOfFeatureVecsTrain = new ArrayBuffer[Array[Double]]()
    var ArrayOfFeatureVecsTest = new ArrayBuffer[Array[Double]]()

    while(iter.hasNext) {
      val x = iter.next()
      columnIndices += x.index
      if(trainingAndTestIndices == null){
//        val feature = DenseVector(x.observations.toArray)
//        val featureMean = sum(feature)/nObs.toDouble
//        val featureSD = math.pow(norm(feature - DenseVector.fill(nObs)(featureMean)),2)/(nObs.toDouble - 1)
//        val centeredAndScaled = (feature - DenseVector.fill(nObs)(featureMean))/featureSD
//        ArrayOfFeatureVecsTrain += centeredAndScaled.toArray
        ArrayOfFeatureVecsTrain += x.observations.toArray

      }else{
        val currentFeatureVector = DenseVector(x.observations.toArray)
        ArrayOfFeatureVecsTrain += currentFeatureVector(trainingAndTestIndices._1).toArray
        ArrayOfFeatureVecsTest+=currentFeatureVector(trainingAndTestIndices._2).toArray
      }
    }

    val n = if(trainingAndTestIndices == null) nObs else trainingAndTestIndices._1.length
    val localMatTrain = new DenseMatrix(n, columnIndices.length, ArrayOfFeatureVecsTrain.toArray.flatten)

    val localMatTest =
      if(trainingAndTestIndices != null){
        Some(new DenseMatrix(trainingAndTestIndices._2.length, columnIndices.length,
          ArrayOfFeatureVecsTest.toArray.flatten))
      }else{
        None
      }

    Iterator((partitionIndex, (columnIndices.toList, localMatTrain, localMatTest)))
  }

  def createLocalMatricesSparse(
                           partitionIndex : Int,
                           iter : Iterator[FeatureVectorLP],
                           nObs : Int,
                           trainingAndTestIndices : (List[Int], List[Int]))
  : Iterator[(Int, (List[Int], CSCMatrix[Double], Option[CSCMatrix[Double]]))] = {

    var columnIndices = new mutable.MutableList[Int]()
    val featureVectors = iter.toArray

    val n = if(trainingAndTestIndices == null) nObs else trainingAndTestIndices._1.length
    val nTest =  if(trainingAndTestIndices == null) None else Some(trainingAndTestIndices._2.length)

    val builderTrain = new CSCMatrix.Builder[Double](rows = n, cols = featureVectors.length)
    val builderTest =
      if(trainingAndTestIndices == null){
        None
      }else{
        Some(new CSCMatrix.Builder[Double](rows = nTest.get, cols = featureVectors.length))
      }

    for(colInd <- 0 until featureVectors.length) {
      val x = featureVectors(colInd)
      columnIndices += x.index
      val featureVec = x.observations

      for(ind <- 0 until n){
        val rowInd = if(trainingAndTestIndices == null) ind else trainingAndTestIndices._1(ind)
        if(featureVec(rowInd) != 0){
          builderTrain.add(ind, colInd, featureVec(rowInd))
        }
      }

      if(trainingAndTestIndices != null){
        for(ind <- 0 until nTest.get){
          val rowInd = trainingAndTestIndices._2(ind)
          if(featureVec(rowInd) != 0){
            builderTest.get.add(ind, colInd, featureVec(rowInd))
          }
        }
      }

    }

    val localMatTrain = builderTrain.result()

    val localMatTest =
      if(trainingAndTestIndices != null){
        Some(builderTest.get.result())
      }else{
        None
      }

    Iterator((partitionIndex, (columnIndices.toList, localMatTrain, localMatTest)))
  }

  def createLocalMatrices(
                           parsedDataByCol : RDD[FeatureVectorLP],
                           useSparseStructure : Boolean,
                           nObs : Int,
                           trainingAndTestIndices : (List[Int], List[Int])
                           ) : RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))] = {

    val localMats =
      if (useSparseStructure) {
        parsedDataByCol.mapPartitionsWithIndex((partitionID, iterator) =>
          createLocalMatricesSparse(partitionID, iterator, nObs, trainingAndTestIndices),
          preservesPartitioning = true
        )
      } else {
        parsedDataByCol.mapPartitionsWithIndex((partitionID, iterator) =>
          createLocalMatricesDense(partitionID, iterator, nObs, trainingAndTestIndices),
          preservesPartitioning = true
        )
      }

    localMats.asInstanceOf[RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))]]
  }


}

