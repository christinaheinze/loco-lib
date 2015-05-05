package LOCO.utils

//
// A simple Scala interface to FFTWLibrary.java for real transforms of arbitrary dimension.
// The object FFTReal demonstrates example usage.
// Author: Kipton Barros
//

import com.sun.jna._
import java.nio.IntBuffer
import fftw3.{FFTW3Library => FFTW}
import FFTW.{INSTANCE => fftw}

object FFTReal {
  
  def test() {
    val dim = Array(3, 3)
    val a = Array[Double](0,0,1, 0,0,0, 0,0,0)
    val b = Array[Double](1,2,3, 4,5,6, 7,8,9)
    val dst = new Array[Double](dim.product)
    val fft = new FFTReal(dim)
    fft.convolve(a, b, dst)
    dst.foreach(println _)
  }

  def test2() {
    val dim = Array(4, 4)
    val a = Array[Double](0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)
    val fft = new FFTReal(dim)
    val ap = fft.allocFourierArray()
    fft.forwardTransform(a, ap)
    
    val bp = fft.allocFourierArray()
    fft.tabulateFourierArray(bp) { k: Array[Double] =>
      (math.cos(-k(1)), math.sin(-k(1)))
    }
    
    for (i <- ap.indices) {
      println(ap(i) - bp(i))
    }
  }

  def test3() {
    val dim = Array(4, 4)
    val a = Array[Double](0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0)
    val fft = new FFTReal(dim)
    val ap = fft.allocFourierArray()
    fft.forwardTransform(a, ap)
    
    val bp = fft.allocFourierArray()
    fft.tabulateFourierArray(bp) { k: Array[Double] =>
      (math.cos(-k(0)), math.sin(-k(0)))
    }
    
    for (i <- ap.indices) {
      println(ap(i) - bp(i))
    }
  }
}


// Note that arrays are packed in row major order, so the last index is the fastest varying.
// Thus, if indices are computed as (i = Lx*y + x) then one should use dim(Ly, Lx)
class FFTReal(dim: Array[Int], lenOption: Option[Array[Double]] = None, flags: Int = FFTW.FFTW_ESTIMATE) {

  val len = lenOption.getOrElse(dim.map(_.toDouble))
  val rank = dim.size
  
  // number of doubles in real-space array
  val n = dim.product
  // number of doubles in reciprocal space array
  val nrecip = 2*(dim.slice(0,rank-1).product*(dim(rank-1)/2+1)) // fftw compresses last index
  
  val sizeofDouble = 8
  val inBytes  = sizeofDouble*n
  val outBytes = sizeofDouble*nrecip
  
  val in = fftw.fftw_malloc(new NativeLong(inBytes))
  val out = fftw.fftw_malloc(new NativeLong(outBytes))
  val inbuf = in.getByteBuffer(0, inBytes).asDoubleBuffer()
  val outbuf = out.getByteBuffer(0, outBytes).asDoubleBuffer()
  val kind= Array[Int](5)
  val invKind= Array[Int](4)
//  val planForward  = fftw.fftw_plan_dft_r2c(dim.size, IntBuffer.wrap(dim), inbuf, outbuf, flags)
//  val planBackward = fftw.fftw_plan_dft_c2r(dim.size, IntBuffer.wrap(dim), outbuf, inbuf, flags)

  val planForward  = fftw.fftw_plan_r2r(dim.size, IntBuffer.wrap(dim), inbuf, outbuf, IntBuffer.wrap(kind), flags)
  val planBackward = fftw.fftw_plan_r2r(dim.size, IntBuffer.wrap(dim), inbuf, outbuf, IntBuffer.wrap(invKind), flags)


  def forwardTransform(src: Array[Double], dst: Array[Double]) {
    require(src.size == n)
    require(dst.size == nrecip)
    
    inbuf.clear()
    inbuf.put(src)
    fftw.fftw_execute(planForward)
    outbuf.rewind()
    outbuf.get(dst)
    
    // continuum normalization: f(k) = \int dx^d f(x) e^(i k x)
    val scale = len.product / dim.product
    for (i <- dst.indices) dst(i) *= scale
  }
  
  def backwardTransform(src: Array[Double], dst: Array[Double]) {
    require(src.size == nrecip)
    require(dst.size == n)
    
    outbuf.clear()
    outbuf.put(src)
    fftw.fftw_execute(planBackward)
    inbuf.rewind()
    inbuf.get(dst)

    // continuum normalization: f(x) = (2 Pi)^(-d) \int dk^d f(k) e^(- i k x)
    val scale = 1 / len.product
    for (i <- dst.indices) dst(i) *= scale
  }
  
  def allocFourierArray(): Array[Double] = {
    new Array[Double](nrecip)
  }
  
  def tabulateFourierArray(dst: Array[Double])(f: Array[Double] => (Double, Double)) {
    require(dst.size == nrecip)
    for (i <- 0 until dst.size/2) {
      val k = fourierVector(i)
      val (re, im) = f(k)
      dst(2*i+0) = re
      dst(2*i+1) = im
    }
  }
  
  def multiplyFourierArrays(src1: Array[Double], src2: Array[Double], dst: Array[Double]) {
    require(src1.size == nrecip)
    require(src2.size == nrecip)
    require(dst.size == nrecip)
    for (i <- 0 until src1.size/2) {
      // src and dst arrays might be aliased; create temporary variables
      val re = src1(2*i+0)*src2(2*i+0) - src1(2*i+1)*src2(2*i+1)
      val im = src1(2*i+0)*src2(2*i+1) + src1(2*i+1)*src2(2*i+0)
      dst(2*i+0) = re
      dst(2*i+1) = im
    }
  }

  def conjugateFourierArray(src: Array[Double], dst: Array[Double]) {
    require(src.size == nrecip)
    require(dst.size == nrecip)
    for (i <- 0 until src.size/2) {
      dst(2*i+0) = src(2*i+0)
      dst(2*i+1) = -src(2*i+1)
    }
  }
  
  // Returns the list of all fourier vectors
  def fourierVectors: Array[Array[Double]] = {
    Array.tabulate(nrecip/2) { fourierVector(_) }
  }
  
  // for each indexed complex number in fourier array, return corresponding vector k
  // where component k(r) = n (2 pi / L_r) for integer n in range [-N/2, +N/2)
  def fourierVector(i: Int): Array[Double] = {
    require(0 <= i && i < nrecip/2)
    val k = new Array[Double](rank)
    var ip = i
    for (r <- rank-1 to 0 by -1) {
      val d = if (r == rank-1) (dim(r)/2+1) else dim(r) // fftw compresses last index
      k(r) = ip % d
      if (k(r) >= dim(r)/2)
        k(r) -= dim(r)
      val dk = 2*math.Pi/len(r)
      k(r) *= dk
      ip /= d
    }
    k
  }
  
  def destroy {
    fftw.fftw_destroy_plan(planForward)
    fftw.fftw_destroy_plan(planBackward)
    fftw.fftw_free(in)
    fftw.fftw_free(out)
  }

  
  def convolve(a: Array[Double], b: Array[Double], dst: Array[Double]) {
    require(a.size == n && b.size == n && dst.size == n)
    val ap = allocFourierArray()
    val bp = allocFourierArray()
    forwardTransform(a, ap)
    forwardTransform(b, bp)
    // conjugateFourierArray(bp, bp) // affects sign: c(j) = \sum_i a(i) b(i-j)
    multiplyFourierArrays(ap, bp, ap)
    backwardTransform(ap, dst)
  }

  def convolveWithRecip(a: Array[Double], dst: Array[Double])(bp: Array[Double]) {
    require(a.size == n && bp.size == nrecip && dst.size == n)
    val ap = allocFourierArray()
    forwardTransform(a, ap)
    multiplyFourierArrays(ap, bp, ap)
    backwardTransform(ap, dst)
  }

  def convolveWithRecipFn(a: Array[Double], dst: Array[Double])(fn: Array[Double] => (Double, Double)) {
    require(a.size == n && dst.size == n)
    val ap = allocFourierArray()
    val bp = allocFourierArray()
    forwardTransform(a, ap)
    tabulateFourierArray(bp)(fn)
    multiplyFourierArrays(ap, bp, ap)
    backwardTransform(ap, dst)
  }
}
