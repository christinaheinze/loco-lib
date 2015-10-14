#!/bin/bash

$SPARK_HOME/bin/spark-submit \
--class "preprocessingUtils.main" \
--master local[4] \
target/scala-2.10/preprocess-assembly-0.2.jar \
--dataFormat=text \
--sparse=false \
--textDataFormat=spaces \
--separateTrainTestFiles=false \
--proportionTest=0.2 \
--dataFile="../data/dogs_vs_cats_n5000.txt" \
--centerFeatures=true \
--scaleFeatures=true \
--centerResponse=false \
--scaleResponse=false \
--outputTrainFileName="../data/dogs_vs_cats_small_train" \
--outputTestFileName="../data/dogs_vs_cats_small_test" \
--outputClass=LabeledPoint \
--seed=1
"$@"