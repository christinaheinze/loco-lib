package LOCO.utils

import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{Matrix, CSCMatrix, DenseMatrix, DenseVector}

import org.apache.spark.rdd.RDD

import preprocessingUtils.FeatureVector
import LOCO.utils.LOCOUtils._


object ProjectionUtils {

  /**
   * Project each workers data matrix and add to RDD.
   * 
   * @param parsedDataByCol Data matrix parsed by column as RDD containing FeatureVectors
   * @param projection Specify which projection shall be used: "sparse" for a sparse random 
   *                   projection or "SDCT" for the SDCT. Note that the latter is not threadsafe!
   * @param flagFFTW flag for SDCT/FFTW: 64 corresponds to FFTW_ESTIMATE,
   *                 0 corresponds to FFTW_MEASURE
   * @param concatenate True if random features should be concatenated.
   *                    When set to false they are added.
   * @param nFeatsProj Dimensionality of the random projection
   * @param nObs Number of observations
   * @param nFeats Number of features
   * @param seed Random seed
   * @param nPartitions Number of partitions used for parsedDataByCol. 
   *                 
   * @return RDD containing the partitionID as key and the column indices, the raw and random 
   *         features as value.                
   */
  def project(
      parsedDataByCol : RDD[FeatureVector],
      projection : String,
      flagFFTW : Int,
      useSparseStructure : Boolean,
      concatenate : Boolean,
      nFeatsProj : Int,
      nObs : Int,
      nFeats : Int,
      seed : Int,
      nPartitions : Int) : RDD[(Int, (List[Int], Matrix[Double], Matrix[Double]))] = {

    // check whether projection dimension has been chosen smaller
    // than number of raw features per worker
    if(concatenate || projection == "SDCT"){
      assert(isValidProjectionDim(nFeatsProj, nPartitions, nFeats),
        "Projection Dimension needs to be smaller than number of raw features " +
          "per partition (= number of features / number of partitions)")
    }

    val res = if(useSparseStructure){
      projectSparse(
        parsedDataByCol, projection, flagFFTW, concatenate, nFeatsProj, nObs, nFeats,
        seed, nPartitions)
    }else{
      projectDense(
        parsedDataByCol, projection, flagFFTW, concatenate, nFeatsProj, nObs, nFeats,
        seed, nPartitions)
    }

    res.asInstanceOf[RDD[(Int, (List[Int], Matrix[Double], Matrix[Double]))]]
  }

  def projectDense(
               parsedDataByCol : RDD[FeatureVector],
               projection : String,
               flagFFTW : Int,
               concatenate : Boolean,
               nFeatsProj : Int,
               nObs : Int,
               nFeats : Int,
               seed : Int,
               nPartitions : Int) : RDD[(Int, (List[Int], DenseMatrix[Double], DenseMatrix[Double]))] = {

    // get partition index and create local raw feature matrices
    val localMats : RDD[(Int, (List[Int], DenseMatrix[Double]))] =
      parsedDataByCol.mapPartitionsWithIndex((partitionID, iterator) =>
        preprocessing.createLocalMatrices(partitionID, iterator, nObs),
        preservesPartitioning = true
      )

    // compute random projections and return resulting RDD
    localMats.mapValues{case(colIndices, rawFeats) =>
      val RP = projection match{
        case "SDCT" => SDCT(rawFeats, nFeatsProj, seed, flagFFTW)
        case "sparse" => rawFeats * sparseProjMat(rawFeats.cols, nFeatsProj, seed)
        case _ => throw new IllegalArgumentException("Invalid argument for projection : " + projection)
      }
      (colIndices, rawFeats, RP)
    }
  }

  def projectSparse(
                    parsedDataByCol : RDD[FeatureVector],
                    projection : String,
                    flagFFTW : Int,
                    concatenate : Boolean,
                    nFeatsProj : Int,
                    nObs : Int,
                    nFeats : Int,
                    seed : Int,
                    nPartitions : Int) : RDD[(Int, (List[Int], CSCMatrix[Double], CSCMatrix[Double]))] = {


    // get partition index and create local raw feature matrices
    val localMats : RDD[(Int, (List[Int], CSCMatrix[Double]))] =
      parsedDataByCol.mapPartitionsWithIndex((partitionID, iterator) =>
        preprocessing.createLocalMatricesSparse(partitionID, iterator, nObs),
        preservesPartitioning = true
      )

    // compute random projections and return resulting RDD
    localMats.mapValues{case(colIndices, rawFeats) =>
      val RP = projection match{
//        case "SDCT" => SDCT(rawFeats, nFeatsProj, seed, flagFFTW)
        case "sparse" =>   rawFeats * sparseProjMatCSC(rawFeats.cols, nFeatsProj, seed)
        case _ => throw new IllegalArgumentException("Invalid argument for projection : " + projection)
      }
      (colIndices, rawFeats, RP)
    }

  }

  /**
   * Sums or concatenates random projections from remaining partitions locally.
   * 
   * @param matrixWithIndex Tuple containing the partition ID as key, and the column indices and
   *                        the raw feature matrix as value.
   * @param randomProjectionMap Map containing the random projection of the remaining partitions.
   * @param concatenate If true, random projections are concatenated, otherwise they are added.
   *
   * @return Returns DenseMatrix with random features as they'll enter in the local design matrix.
   */
  def randomFeaturesConcatenateOrAddAtWorker(
      matrixWithIndex: (Int, (List[Int], Matrix[Double])),
      randomProjectionMap : collection.Map[Int, Matrix[Double]],
      concatenate : Boolean) : Matrix[Double] = {
    
    // extract key of worker
    val key = matrixWithIndex._1
    
    // exclude own random features from RPsMap
    val randomProjectionMapFiltered = randomProjectionMap.filter(x => x._1 != key)
    
    // concatenate or sum random projection matrices from remaining workers
    aggregationOfRPs(randomProjectionMapFiltered.values, concatenate)
  }


  /**
   * Subtracts own random projection locally.
   * 
   * @param matrixWithIndex Tuple containing the partition ID as key, and the column indices, 
   *                        the raw feature matrix and the random feature matrix as value.
   * @param randomProjectionsAdded DenseMatrix containing the sum of all random projections.
   *
   * @return Returns DenseMatrix with random features as they'll enter in the local design matrix.
   */
  def randomFeaturesSubtractLocal(
      matrixWithIndex: (Int, (List[Int], Matrix[Double], Matrix[Double])),
      randomProjectionsAdded : Matrix[Double]) : Matrix[Double] = {
    randomProjectionsAdded - matrixWithIndex._2._3
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

    (0 until nFeatures).map { i =>
      (0 until nProjDim).map { j =>
        val entry = sample(dist)
        if(!(entry == 0)){
          builder.add(i, j, math.sqrt(3.0/nProjDim) * entry)
        }
      }
    }

    builder.result()
  }


  /**
   * Projects the input matrix with a sparse random projection.
   *
   * @param input Input matrix
   * @param nFeatures Number of features in input matrix
   * @param nProjDim Projection dimension
   * @param seed Random seed
   *
   * @return Projected input matrix
   */
  def sparseRP(
      input : DenseMatrix[Double],
      nFeatures : Int,
      nProjDim : Int,
      seed : Int) : DenseMatrix[Double] = {
    // multiply input with sparse random projection matrix
    input * sparseProjMat(nFeatures, nProjDim, seed)
  }

  /**
   * Computes the discrete cosine tranfrom of input matrix.
   *
   * @param dataMat Input matrix
   * @param nProjDim Projection dimension
   * @param diagonal
   * @param cols Column-wise projection if cols is set to true, otherwise row-wise projection
   * @param flagFFTW flag for SDCT/FFTW: 64 corresponds to FFTW_ESTIMATE,
   *                 0 corresponds to FFTW_MEASURE
   *
   * @return Projected input matrix
   */
  def DCT(
      dataMat : DenseMatrix[Double],
      nProjDim : Int,
      diagonal : DenseVector[Double],
      cols : Boolean,
      flagFFTW : Int) : DenseMatrix[Double] = {

    // extract the dimension of the vectors that should be transformed
    // and the number of vectors that should be transformed
    val dimOfVecsToTransform = if(cols) dataMat.cols else dataMat.rows
    val nVecsToTransform = if(cols) dataMat.rows else dataMat.cols

    // scaling
    val scale = 1 / math.sqrt(2 * dimOfVecsToTransform)

    // initializations
    val dim = Array(dimOfVecsToTransform, 1)
    val fft = new FFTReal(dim, flags = flagFFTW)
    var ArrayOfFeatureVecs = new ArrayBuffer[Array[Double]]()

    // transform
    for(i <- 0 until nVecsToTransform){
      val vec : Array[Double] =
            if(cols) (diagonal :* dataMat(i, ::).t * scale).toArray
            else (diagonal :* dataMat(::, i) * scale).toArray

      var dest = fft.allocFourierArray()
      fft.forwardTransform(vec, dest)
      ArrayOfFeatureVecs += dest.slice(0, dimOfVecsToTransform)
    }

    // transpose back if needed
    if(cols){
      new DenseMatrix(dimOfVecsToTransform, nVecsToTransform, ArrayOfFeatureVecs.toArray.flatten).t
    }else{
      new DenseMatrix(dimOfVecsToTransform, nVecsToTransform, ArrayOfFeatureVecs.toArray.flatten)
    }
  }

  /**
   * Computes the subsampled randomized DCT of input matrix.
   *
   * @param dataMat Input matrix
   * @param nProjDim Projection dimension
   * @param seed Random seed
   * @param flagFFTW flag for SDCT/FFTW: 64 corresponds to FFTW_ESTIMATE,
   *                 0 corresponds to FFTW_MEASURE
   * @param cols Column-wise projection if cols is set to true, otherwise row-wise projection
   *
   * @return Projected input matrix
   */
  def SDCT(
      dataMat : DenseMatrix[Double],
      nProjDim : Int,
      seed : Int,
      flagFFTW : Int,
      cols : Boolean = true) : DenseMatrix[Double]= {

    // number of observations
    val n = dataMat.rows
    // number of features
    val p = dataMat.cols

    // dimension to be compressed
    val dim = if(cols) p else n

    // compute scaling factor
    val srhtConst = math.sqrt(dim / nProjDim.toDouble)

    // sample from Rademacher distribution and compute diagonal
    val dist = Map(-1.0 -> 1.toDouble/2, 1.0 -> 1.toDouble/2)
    val D = DenseVector.tabulate(dim){i => sample(dist)} * srhtConst

    // compute the DCT
    var res = DCT(dataMat, nProjDim, D, cols, flagFFTW)

    // subsample
    val subsampledIndices = util.Random.shuffle(List.fill(1)(0 until dim).flatten).take(nProjDim)

    res = res(::, subsampledIndices).toDenseMatrix

    // transpose back if needed
    if(cols){
      res
    }else{
      res.t
    }
  }

}
