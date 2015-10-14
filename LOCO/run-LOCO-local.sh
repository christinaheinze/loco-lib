#!/bin/bash

$SPARK_HOME/bin/spark-submit \
--class "LOCO.driver" \
--master local[4] \
--driver-memory 1G \
target/scala-2.10/LOCO-assembly-0.2.0.jar \
--classification=false \
--numIterations=5000 \
--trainingDatafile="../data/climate-serialized/climate-train-colwise/" \
--testDatafile="../data/climate-serialized/climate-test-colwise/" \
--responsePathTrain="../data/climate-serialized/climate-responseTrain.txt" \
--responsePathTest="../data/climate-serialized/climate-responseTest.txt" \
--nFeats="../data/climate-serialized/climate-nFeats.txt" \
--projection=SDCT \
--concatenate=false \
--CV=false \
--lambda=75 \
--nFeatsProj=389 \
--nPartitions=4 \
--nExecutors=1
"$@"