package LOCO.solvers

import breeze.linalg._
import preprocessingUtils.utils.metrics._


object SDCA {
  /**
   * Runs SDCA on local data matrix.
   *
   * Note that SDCA for hinge-loss is equivalent to LibLinear, where using the
   * regularization parameter  C = 1.0/(lambda*numExamples), and re-scaling
   * the alpha variables with 1/C.
   *
   * @param localData Local data examples
   * @param response Response vector
   * @param localIters number of local coordinates to update
   * @param lambda Regularization parameter
   * @param n Number of observations
   * @param seed Random seed
   * @param checkDualityGap Specify whether the duality gap should be computed after each iteration.
   *                        Note that this is expensive as it requires a pass over the entire (local)
   *                        data set. Should only be used for tuning purposes.
   * @param stoppingDualityGap Specify the size of the duality gap at which the optimization should
   *                           end. If it is not reached after numIterations, the optimization ends
   *                           nonetheless.
   *
   * @return alpha - dual solution vector
   */
  def localSDCA(
      localData: DenseMatrix[Double],
      response : Vector[Double],
      localIters: Int,
      lambda: Double,
      n: Int,
      p : Int,
      seed: Int,
      checkDualityGap : Boolean,
      stoppingDualityGap : Double): Vector[Double] = {

    var alpha = Vector.fill(n)(0.0)
    var w = Vector.fill(p)(0.0)
    val r = new scala.util.Random(seed)

    var it = 0

    val checkCondition = (checkDualityGap : Boolean) => {
      if(checkDualityGap)
        if(
          computeDualityGap(localData, response, n, w, alpha, lambda,
            hingeLossPrimal, hingeLossDual) > stoppingDualityGap
        )
          true
        else
          false
      else
        true
    }

    // perform local updates
    for (i <- 1 to localIters if checkCondition(checkDualityGap)){
      it = i

      // randomly select a local example
      val idx = r.nextInt(n)
      val y = response(idx)
      val x : breeze.linalg.Vector[Double] = localData(idx, ::).t

      // compute hinge loss gradient
      val grad = (y*(x dot w) - 1.0)*(lambda*n)

      // compute projected gradient
      var proj_grad = grad
      if (alpha(idx) <= 0.0)
        proj_grad = Math.min(grad,0)
      else if (alpha(idx) >= 1.0)
        proj_grad = Math.max(grad,0)

      if (Math.abs(proj_grad) != 0.0 ) {
        val qii  = x dot x
        var newAlpha = 1.0
        if (qii != 0.0) {
          newAlpha = Math.min(Math.max((alpha(idx) - (grad / qii)), 0.0), 1.0)
        }

        // update primal and dual variables
        val update = x*( y*(newAlpha-alpha(idx))/(lambda*n) )
        w = update + w
        alpha(idx) = newAlpha
      }
    }

    if(checkDualityGap)
      println("Duality gap: " +
        computeDualityGap(localData, response, n, w, alpha, lambda, hingeLossPrimal, hingeLossDual)
        + " reached in iteration " + it)

   // return alpha
    alpha :* response
  }


  def localSDCARidge(localData: DenseMatrix[Double],
                response : Vector[Double],
                nIterations: Int,
                lambda: Double,
                n: Int,
                p : Int,
                seed: Int,
                checkDualityGap : Boolean,
                stoppingDualityGap : Double): Vector[Double] = {

    var alpha = Vector.fill(n)(0.0)
    var w: DenseVector[Double] = DenseVector.fill(p)(0.0)
    val r = new scala.util.Random(seed)

    var it = 0

    val checkCondition = (checkDualityGap : Boolean) => {
      if(checkDualityGap)
        if(
          computeDualityGap(localData, response, n, w, alpha, lambda,
            squaredLossPrimal, squaredLossDual) > stoppingDualityGap
        )
          true
        else
          false
      else
        true
    }

    // perform local updates
    for (i <- 1 to nIterations if checkCondition(checkDualityGap)){
      it = i

      // randomly select a local example
      val idx = r.nextInt(n)
      val y = response(idx)
      val x: DenseVector[Double] = localData(idx, ::).t

      // delta alpha
      val deltaAlpha = (y - (x dot w) - 0.5*alpha(idx))/(0.5 + (x dot x)/(lambda*n))

      // update primal and dual variables
      w = w + x * deltaAlpha/(lambda*n)
      alpha(idx) = alpha(idx) + deltaAlpha

    }

    if(checkDualityGap)
      println("Duality gap: " +
        computeDualityGap(localData, response, n, w, alpha, lambda,
          squaredLossPrimal, squaredLossDual)
        + " reached in iteration " + it)

    // return alpha
    alpha
  }


  def localSDCALogistic(localData: DenseMatrix[Double],
                         response : Vector[Double],
                         nIterations: Int,
                         lambda: Double,
                         n: Int,
                         p : Int,
                         seed: Int,
                         checkDualityGap : Boolean,
                         stoppingDualityGap : Double): Vector[Double] = {

    var alpha = Vector.fill(n)(0.0)
    var w = Vector.fill(p)(0.0)
    val r = new scala.util.Random(seed)

    var it = 0

    val checkCondition = (checkDualityGap : Boolean) => {
      if(checkDualityGap)
        if(
          computeDualityGap(localData, response, n, w, alpha, lambda,
            squaredLossPrimal, squaredLossDual) > stoppingDualityGap
        )
          true
        else
          false
      else
        true
    }

    // perform local updates
    for (i <- 1 to nIterations if checkCondition(checkDualityGap)){
      it = i

      // randomly select a local example
      val idx = r.nextInt(n)
      val y = response(idx)
      val x = localData(idx, ::).t

      // delta alpha
      val deltaAlpha = (y/(1 + math.exp(y * (x dot w))) - alpha(idx))/math.max(1, 0.25 + (x dot x)/(lambda*n))

      // update primal and dual variables
      w = w + x * deltaAlpha/(lambda*n)
      alpha(idx) = alpha(idx) + deltaAlpha

    }

    if(checkDualityGap)
      println("Duality gap: " +
        computeDualityGap(localData, response, n, w, alpha, lambda,
          squaredLossPrimal, squaredLossDual)
        + " reached in iteration " + it)

    // return alpha
    alpha
  }

}

