package preprocessingUtils.preprocess

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object transform {

  /**
   * Center and/or scale the features and/or the response vector.
   *
   * @param data Data provided as RDD containing LabeledPoints.
   * @param centerFeatures True if features are to be centered.
   * @param centerResponse True if response is to be centered.
   * @param scaleFeatures True if features are to be scaled to have unit variance.
   * @param scaleResponse True if response is to be scaled to have unit variance.
   * @return Centered/scaled data as RDD containing LabeledPoints.
   */
  def centerAndScale(
      data : RDD[LabeledPoint],
      centerFeatures : Boolean,
      centerResponse : Boolean,
      scaleFeatures : Boolean,
      scaleResponse : Boolean) : RDD[LabeledPoint] = {

    // create StandardScaler for features
    val scaler_features =
      if(centerFeatures || scaleFeatures)
        new StandardScaler(withMean = centerFeatures, withStd = scaleFeatures)
          .fit(data.map(x => x.features))
      else
        null

    // create StandardScaler for response
    val scaler_response =
      if(centerResponse || scaleResponse)
        new StandardScaler(withMean = centerResponse, withStd = scaleResponse)
          .fit(data.map(x => Vectors.dense(x.label)))
      else
        null

    if(centerFeatures || scaleFeatures || centerResponse || scaleResponse)
      // apply to data
      data.map(x => LabeledPoint(
        if(centerResponse || scaleResponse) {
          scaler_response.transform(Vectors.dense(x.label))(0)
        }else{
          x.label
        },
        if(centerFeatures || scaleFeatures) {
          scaler_features.transform(x.features)
        }else{
          x.features
        }
      ))
    else
      data
  }
}
