package LOCO.utils

import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{Matrix, CSCMatrix, DenseMatrix, DenseVector}
import edu.emory.mathcs.jtransforms.dct._

import org.apache.spark.rdd.RDD

object ProjectionUtils {

  /**
   * Project each workers data matrix and add to RDD.
   * 
   * @param localMats RDD of tuples containing the partition ID as key. The value is in turn a tuple with
   *                  the list of indices of the feature vectors in this partition, the local matrix
   *                  with training observations and, optionally, the local matrix with test
   *                  observations.
   * @param projection Specify which projection shall be used: "sparse" for a sparse random 
   *                   projection or "SDCT" for the SDCT.
   * @param useSparseStructure Set to true if sparse data structures should be used.
   * @param nFeatsProj Dimensionality of the random projection
   * @param nObs Number of observations
   * @param nFeats Number of features
   * @param seed Random seed
   * @param nPartitions Number of partitions used for parsedDataByCol. 
   *                 
   * @return RDD containing the partitionID as key and the column indices, the raw and random 
   *         features as value. Optionally, the value also contains a matrix with raw test
   *         observations.
   */
  def project(
               localMats : RDD[(Int, (List[Int], Matrix[Double], Option[Matrix[Double]]))],
               projection : String,
               useSparseStructure : Boolean,
               nFeatsProj : Int,
               nObs : Int,
               nFeats : Int,
               seed : Int,
               nPartitions : Int)
  : RDD[(Int, (List[Int], Matrix[Double], DenseMatrix[Double], Option[Matrix[Double]]))] = {

    val res = if(useSparseStructure){
      projectSparse(
        localMats.asInstanceOf[RDD[(Int, (List[Int], CSCMatrix[Double], Option[CSCMatrix[Double]]))]],
        projection, nFeatsProj, seed)
    }else{
      projectDense(
        localMats.asInstanceOf[RDD[(Int, (List[Int], DenseMatrix[Double], Option[DenseMatrix[Double]]))]],
        projection, nFeatsProj, seed)
    }

    res.asInstanceOf[RDD[(Int, (List[Int], Matrix[Double], DenseMatrix[Double], Option[Matrix[Double]]))]]
  }

  def projectDense(
                    localMats : RDD[(Int, (List[Int], DenseMatrix[Double], Option[DenseMatrix[Double]]))],
                    projection : String,
                    nFeatsProj : Int,
                    seed : Int)
  : RDD[(Int, (List[Int], DenseMatrix[Double], DenseMatrix[Double], Option[DenseMatrix[Double]]))] = {

    // compute random projections and return resulting RDD
    localMats.mapValues{case(colIndices, rawFeatsTrain, rawFeatsTest) =>

      // check whether projection dimension has been chosen smaller
      // than number of raw features per worker
      if(projection == "SDCT"){
        assert(nFeatsProj <= rawFeatsTrain.cols,
          "Projection Dimension needs to be smaller than number of raw features " +
            "per partition (= number of features / number of partitions)")
      }

      val RP = projection match{
        case "SDCT" => SubsampledDCT(rawFeatsTrain, nFeatsProj, seed)
        case "sparse" => rawFeatsTrain * sparseProjMat(rawFeatsTrain.cols, nFeatsProj, seed)
        case _ => throw new IllegalArgumentException("Invalid argument for projection : " + projection)
      }
      (colIndices, rawFeatsTrain, RP, rawFeatsTest)
    }
  }

  def projectSparse(
                     localMats : RDD[(Int, (List[Int], CSCMatrix[Double], Option[CSCMatrix[Double]]))],
                     projection : String,
                     nFeatsProj : Int,
                     seed : Int)
  : RDD[(Int, (List[Int], CSCMatrix[Double], DenseMatrix[Double], Option[CSCMatrix[Double]]))] = {


    // compute random projections and return resulting RDD
    localMats.mapValues{case(colIndices, rawFeatsTrain, rawFeatsTest) =>
      val RP = projection match{
        case "SDCT" =>
          SubsampledDCT(rawFeatsTrain, nFeatsProj, seed)
        case "sparse" =>
          (rawFeatsTrain * sparseProjMatCSC(rawFeatsTrain.cols, nFeatsProj, seed)).toDenseMatrix
        case _ =>
          throw new IllegalArgumentException("Invalid argument for projection : " + projection)
      }
      (colIndices, rawFeatsTrain, RP, rawFeatsTest)
    }

  }

  /**
   * Sums or concatenates random projections from remaining partitions locally.
   * 
   * @param key Partition ID
   * @param randomProjectionMap Map containing the random projection of the all partitions.
   * @param concatenate If true, random projections are concatenated, otherwise they are added.
   *
   * @return Returns DenseMatrix with random features as they'll enter in the local design matrix.
   */
  def randomFeaturesConcatenateOrAddAtWorker(
      key : Int,
      randomProjectionMap : collection.Map[Int, Matrix[Double]],
      concatenate : Boolean) : Matrix[Double] = {
    
    // exclude own random features from RPsMap
    val randomProjectionMapFiltered = randomProjectionMap.filter(x => x._1 != key)
    
    // concatenate or sum random projection matrices from remaining workers
    aggregationOfRPs(randomProjectionMapFiltered.values, concatenate)
  }


  /**
   * Subtracts own random projection locally.
   * 
   * @param ownLocalRandomFeatures Tuple containing the partition ID as key, and the column indices,
   *                        the raw feature matrix and the random feature matrix as value.
   * @param randomProjectionsAdded DenseMatrix containing the sum of all random projections.
   *
   * @return Returns DenseMatrix with random features as they'll enter in the local design matrix.
   */
  def randomFeaturesSubtractLocal(
      ownLocalRandomFeatures:  Matrix[Double],
      randomProjectionsAdded : Matrix[Double]) : Matrix[Double] = {
    randomProjectionsAdded - ownLocalRandomFeatures
  }

  /**
   * Adds or concatenates the random projections supplied.
   * 
   * @param randomProjectionMap Map containing the random projections from the other partitions.
   * @param concatenate If true, random projections are concatenated, otherwise they are added.
   *
   * @return Returns DenseMatrix with random features as they'll enter in the local design matrix.
   */
  def aggregationOfRPs(
      randomProjectionMap : Iterable[Matrix[Double]],
      concatenate : Boolean) : Matrix[Double] = {

    if(concatenate){
      // concatenate random projection matrices from remaining workers
      randomProjectionMap.reduce((x, y) => DenseMatrix.horzcat(x, y))
    }else{
      // add random projection matrices from remaining workers
      randomProjectionMap.reduce((x, y) => x + y)
    }
  }

  /**
   * Function to sample from Map according to specified probability distribution.
   *
   * @param dist Distribution
   * @tparam A Type parameter
   * @return sample
   */
  def sample[A](dist: Map[A, Double]): A = {
    val p = scala.util.Random.nextDouble
    val it = dist.iterator
    var accum = 0.0
    while (it.hasNext) {
      val (item, itemProb) = it.next
      accum += itemProb
      if (accum >= p)
        return item
    }
    sys.error(f"this should never happen")  // needed so it will compile
  }

  /**
   * Create a sparse RP matrix.
   *
   * @param nFeatures Number of features to be projection
   * @param nProjDim Projection dimension
   * @param seed Random seed
   *
   * @return A sparse random projection matrix.
   */
  def sparseProjMat(
      nFeatures : Int,
      nProjDim : Int,
      seed : Int) : DenseMatrix[Double] = {

    // initialize matrix
    val init_mat = DenseMatrix.zeros[Double](nFeatures, nProjDim)

    // specify sample distribution
    val dist = Map(-1.0 -> 1.toDouble/6, 0.0 -> 2.toDouble/3, 1.0 -> 1.toDouble/6)

    // set seed
    scala.util.Random.setSeed(seed)

    // create matrix with entries drawn from dist
    DenseMatrix.tabulate(nFeatures, nProjDim){case (i, j) => math.sqrt(3.0/nProjDim) * sample(dist)}
  }

  /**
   * Create a sparse RP matrix using CSCMatrix .
   *
   * @param nFeatures Number of features to be projection
   * @param nProjDim Projection dimension
   * @param seed Random seed
   *
   * @return A sparse random projection matrix.
   */
  def sparseProjMatCSC(nFeatures : Int, nProjDim : Int, seed : Int) : CSCMatrix[Double] = {

    // specify sample distribution
    val dist = Map(-1.0 -> 1.toDouble/6, 0.0 -> 2.toDouble/3, 1.0 -> 1.toDouble/6)

    // set seed
    scala.util.Random.setSeed(seed)

    // initialize builder
    val builder = new CSCMatrix.Builder[Double](rows = nFeatures, cols = nProjDim)

    for(i <- 0 until nFeatures){
      for(j <- 0 until nProjDim){
        val entry = sample(dist)

        if(entry != 0.0){
          builder.add(i, j, math.sqrt(3.0/nProjDim) * entry)
        }
      }
    }

    builder.result()
  }

  /*
    Discrete cosine transform
   */
  def DCTjTrans(vec : Array[Double]) : Array[Double] = {
      val result : Array[Double] = vec.toArray
      val jTransformer = new DoubleDCT_1D(result.length)
      jTransformer.forward(result, true)
      result
  }

  /*
    Subsampled discrete cosine transform
   */
  def SubsampledDCT(
                     dataMat : Matrix[Double],
                     nProjDim : Int,
                     seed : Int
                     ) : DenseMatrix[Double]= {

    // set seed
    scala.util.Random.setSeed(seed)

    // number of observations
    val n = dataMat.rows

    // number of features
    val dim = dataMat.cols

    // buffer for transformed vectors
    var ArrayOfFeatureVecs = new ArrayBuffer[Array[Double]]()

    // compute scaling factor
    val srhtConst = math.sqrt(dim / nProjDim.toDouble)

    // sample from Rademacher distribution and compute diagonal
    val dist = Map(-1.0 -> 1.toDouble/2, 1.0 -> 1.toDouble/2)
    val D : DenseVector[Double] = DenseVector.tabulate(dim){i => sample(dist)} * srhtConst

    // transform
    for(i <- 0 until n){
      val rowVector = dataMat(i to i, 0 until dim).toDenseMatrix.toDenseVector
      val vec : Array[Double] = (D :* rowVector).toArray
      val tmp = DCTjTrans(vec)
      ArrayOfFeatureVecs += tmp
    }

    val res = new DenseMatrix(dim, n, ArrayOfFeatureVecs.toArray.flatten).t

    // subsample
    val subsampledIndices = util.Random.shuffle(List.fill(1)(0 until dim).flatten).take(nProjDim)

    res(::, subsampledIndices).toDenseMatrix
  }
}
