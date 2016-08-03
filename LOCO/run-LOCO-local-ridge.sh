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
--trainingDatafile="../data/climate-serialized/climate-train-colwise/" \
--testDatafile="../data/climate-serialized/climate-test-colwise/" \
--responsePathTrain="../data/climate-serialized/climate-responseTrain.txt" \
--responsePathTest="../data/climate-serialized/climate-responseTest.txt" \
--nFeats="../data/climate-serialized/climate-nFeats.txt" \
--useSparseStructure=false \
--classification=false \
--numIterations=5000 \
--projection=SDCT \
--concatenate=false \
--CV=false \
--lambda=75 \
--nFeatsProj=389 \
--seed=3
"$@"