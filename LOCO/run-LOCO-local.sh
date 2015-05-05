#!/bin/bash

$SPARK_HOME/bin/spark-submit \
--class "LOCO.driver" \
--master local[4] \
--driver-memory 1G \
target/scala-2.10/LOCO-assembly-0.1.jar \
--classification=false \
--optimizer=SDCA \
--numIterations=5000 \
--dataFormat=text \
--textDataFormat=spaces \
--separateTrainTestFiles=true \
--trainingDatafile="../data/climate_train.txt" \
--testDatafile="../data/climate_test.txt" \
--center=true \
--Proj=sparse \
--concatenate=true \
--CVKind=none \
--lambda=70 \
--nFeatsProj=260 \
--nPartitions=4 \
--nExecutors=1
"$@"