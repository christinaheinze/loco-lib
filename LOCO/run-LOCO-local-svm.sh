#!/bin/bash

$SPARK_HOME/bin/spark-submit \
--class "LOCO.driver" \
--master local[4] \
--driver-memory 1G \
target/scala-2.10/LOCO-assembly-0.3.0.jar \
--outdir="output" \
--saveToHDFS=false \
--readFromHDFS=false \
--nPartitions=4 \
--nExecutors=1 \
--trainingDatafile="../data/dogs_vs_cats/dogs_vs_cats_small_train-colwise/" \
--testDatafile="../data/dogs_vs_cats/dogs_vs_cats_small_test-colwise/" \
--responsePathTrain="../data/dogs_vs_cats/dogs_vs_cats_small_train-responseTrain.txt" \
--responsePathTest="../data/dogs_vs_cats/dogs_vs_cats_small_test-responseTest.txt" \
--nFeats="../data/dogs_vs_cats/dogs_vs_cats_small_train-nFeats.txt" \
--useSparseStructure=false \
--classification=true \
--numIterations=5000 \
--projection=SDCT \
--concatenate=false \
--CV=false \
--lambda=4.4 \
--nFeatsProj=200 \
--seed=2
"$@"