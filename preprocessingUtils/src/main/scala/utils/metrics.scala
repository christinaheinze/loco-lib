package preprocessingUtils.utils

import breeze.linalg.{DenseMatrix, sum, Vector}
import org.apache.spark.rdd.RDD

object metrics {

  /**
   *  Computes the mean square error, standardized by the squared norm of the centered response
   */
  def compute_standardizedMSE[T](coefficientVector : Vector[Double], data : RDD[T])
                                (implicit extract : T => (Double, Vector[Double])) : Double = {

    val valuesAndPreds : RDD[(Double, Double)] =
      data.map { point =>
        val (label, features) = extract(point)
        (label, features dot coefficientVector)
      }.cache()

    val num : Double =
      valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()

    val meanResponse =
      data.map{ x =>
        val (label, features) = extract(x)
        label
      }.mean()

    val denom : Double =
      valuesAndPreds.map{case(v, p) => math.pow(v - meanResponse, 2)}.mean()

    num/denom
  }

  /** Computes the mean square error */
  def compute_MSE[T](coefficientVector : Vector[Double], data : RDD[T])
                    (implicit extract : T => (Double, Vector[Double])) : Double = {

    val valuesAndPreds : RDD[(Double, Double)] =
      data.map { point =>
        val (label, features) = extract(point)
        (label, features dot coefficientVector)
      }

    valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
  }

  /** Computes the classification error */
  def computeClassificationError[T](coefficientVector : Vector[Double], data : RDD[T])
                                   (implicit extract : T => (Double, Vector[Double])) : Double = {

    data.map{point : T =>
      val (label, features) = extract(point)
      if((features dot coefficientVector * label) > 0.0) 0.0 else 1.0
    }.mean()
  }

  /** Calculate hinge loss given a point (label,features) and a (primal) weight vector */
  def hingeLossPrimal(label: Double, features : Vector[Double], w: Vector[Double]) : Double = {
    Math.max(1 - (features dot w) * label, 0.0)
  }

  /** Calculate squared loss given a point (label,features) and a (primal) weight vector */
  def squaredLossPrimal(label: Double, features : Vector[Double], w: Vector[Double]) : Double = {
   Math.pow(label - (features dot w), 2)
  }

  /** Calculate conjugate of hinge loss given a dual weight vector  */
  def hingeLossDual(alpha: Vector[Double], response : Vector[Double]) : Double = {
    - sum(alpha)
  }

  /** Calculate conjugate of squared loss given a dual weight vector and the response */
  def squaredLossDual(alpha: Vector[Double], response : Vector[Double]) : Double = {
    var sum = 0.0
    for(i <- 0 until alpha.length){
      sum = sum + 0.25 * Math.pow(alpha(i), 2) - response(i) * alpha(i)
    }
    sum
  }

  /**
   * Compute the average loss over the data set given a loss function f, primal variables w
   * and the response.
   */
  def computeAvgLoss(
      data: DenseMatrix[Double],
      response : Vector[Double],
      n : Int,
      w: Vector[Double],
      f : (Double, Vector[Double], Vector[Double]) => Double) : Double = {

    var loss = 0.0
    for(i <- 0 until data.rows){
      loss = loss + f(response(i), data(i, ::).t, w)
    }
    loss/n
  }

  /** Compute the primal objective value for given w (averaged over n data points). */
  def computePrimalObjective(
      data: DenseMatrix[Double],
      response : Vector[Double],
      n : Int,
      w: Vector[Double],
      lambda: Double,
      f : (Double, Vector[Double], Vector[Double]) => Double): Double = {

    val wVec = w.toDenseVector
    // loss function + l2 regularizer
    computeAvgLoss(data, response, n, w, f) +  lambda * 0.5 * (wVec dot wVec)
  }

  /** Compute the dual objective value for given w and alpha (averaged over n data points). */
  def computeDualObjective(
      n : Int,
      w: Vector[Double],
      alpha : Vector[Double],
      lambda: Double,
      response : Vector[Double],
      f : (Vector[Double], Vector[Double]) => Double): Double = {

    val wVec = w.toDenseVector
    -lambda * 0.5 * (wVec dot wVec) - f(alpha, response) / n
  }

  /**
   * Given primal and dual variables, compute the gap between the primal objective value
   * and the dual objective value.
   */
  def computeDualityGap(
      data : DenseMatrix[Double],
      response : Vector[Double],
      n : Int,
      w: Vector[Double],
      alpha: Vector[Double],
      lambda: Double,
      fPrimal : (Double, Vector[Double], Vector[Double]) => Double,
      fDual : (Vector[Double], Vector[Double]) => Double): Double = {

    computePrimalObjective(data, response, n, w, lambda, fPrimal) - computeDualObjective(n, w,
      alpha, lambda, response, fDual)
  }

}
