package preprocessingUtils

import breeze.linalg.Vector

/**
 * Case class DataPoint for dense label-feature vector pairs, modeled after MLlib's LabeledPoint
 * using breeze.lingalg.Vector */
case class DataPoint(label: Double, features: Vector[Double])

/** Case class FeatureVector for storing a feature vector with an index */
case class FeatureVector(index : Int, observations: Vector[Double])

object DataPoint{
  implicit def extract(x : DataPoint) : (Double, Vector[Double]) = (x.label, x.features)
}


/** Case class FeatureVector for storing a feature vector with an index */
case class FeatureVectorLP(index : Int, observations: org.apache.spark.mllib.linalg.Vector)

object FeatureVectorLP{
  implicit val idxOrdering = new Ordering[Int] {
    override def compare(a: Int, b: Int) = a.compareTo(b)
  }
}