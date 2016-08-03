#!/bin/bash

$SPARK_HOME/bin/spark-submit \
--class "preprocessingUtils.main" \
--master local[4] \
target/scala-2.10/preprocess-assembly-0.3.jar \
--outdir="../data/dogs_vs_cats2/" \
--saveToHDFS=true \
--nPartitions=4 \
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
--outputTrainFileName="dogs_vs_cats_small_train" \
--outputTestFileName="dogs_vs_cats_small_test" \
--twoOutputClasses=false \
--seed=1
"$@"