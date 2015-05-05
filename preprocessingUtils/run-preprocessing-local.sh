#!/bin/bash

$SPARK_HOME/bin/spark-submit \
--class "preprocessingUtils.main" \
--master local[4] \
target/scala-2.10/preprocess-assembly-0.1.jar \
--dataFormat=text \
--textDataFormat=spaces \
--separateTrainTestFiles=false \
--dataFile="../data/dogs_vs_cats_n5000.txt" \
--centerFeatures=true \
--scaleFeatures=true \
--centerResponse=false \
--scaleResponse=false \
--outputTrainFileName="../data/dogs_vs_cats_n5000_train_" \
--outputTestFileName="../data/dogs_vs_cats_n5000_test_" \
--outputClass=DataPoint \
--twoOutputClasses=true \
--secondOutputClass=LabeledPoint
"$@"