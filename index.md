---
layout: dis_template
---

NOTE: THIS PAGE IS STILL UNDER DEVELOPMENT AND THE SOFTWARE HAS NOT BEEN PUBLISHED

# Problem setting

Given a data matrix \\( \mathbf{X} \in \mathbb{R}^{n\times p} \\) and response \\( \mathbf{y} \in \mathbb{R}^n \\), LOCO is a LOw-COmmunication distributed algorithm for \\( \ell_2 \\) - penalised convex estimation problems of the form

\\[ \min_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = \frac{1}{n} \sum\_{i=1}^n f_i(\boldsymbol{\beta}^\top \mathbf{x}_i) + \frac{\lambda}{2} \Vert \boldsymbol{\beta} \Vert^2_2 \\]

LOCO is suitable in a setting where the number of features \\( p \\) is very large so that splitting the design matrix 
across _features_ rather than observations is reasonable. For instance, the number of observations \\( n \\) can be smaller than \\( p \\) or on the same order. LOCO is expected to yield good performance when the rank of the data matrix \\( \boldsymbol{X} \\) is much lower than the actual dimensionality of the observations \\( p \\).

<h1> Table of Contents </h1>
* auto-gen TOC:
{:toc}

# One-shot communication scheme

One large advantage of LOCO over iterative distributed algorithms such as distributed stochastic gradient descent (SGD) is that LOCO only needs one round of communication before the results are sent back to the driver. Therefore, the communication cost -- which is generally the bottleneck in distributed computing -- is very low.

LOCO proceeds as follows. As a preprocessing step, the features need to be distributed across processing units by randomly partitioning the data into K blocks. After this first step, some number of feature vectors are stored on each worker. We shall call these features the "raw" features of worker \\(k\\). Subsequently, each worker applies a dimensionality reduction on its raw features by using a random projection. The resulting features are called the "random features" of worker \\(k\\). Then each worker sends its random features to the other workers. Each worker then adds or concatenates the random features from the other workers and appends these random features to its own raw features. Using this scheme, each worker has access to its raw features and, additionally, to a compressed version of the remaining workers' raw features. Using this design matrix, each worker estimates coefficients locally and returns the ones for its own raw features. As these were learned using the contribution from the random features, they approximate the optimal coefficients sufficiently well. The final estimate returned by LOCO is simply the concatenation of the raw feature coefficients returned from the \\( K \\) workers. 

LOCO's distribution scheme is illustrated in the following figure.

![LOCO]({{ site.baseurl }}/images/locoscheme.png)

# Obtaining the software

**Option 1 - Build with sbt**

Checkout the project repository

	git clone https://github.com/christinaheinze/loco-lib.git

and build the package with
{% highlight bash %}
cd loco-lib/LOCO
sbt assembly
{% endhighlight %}

To install `sbt` on Mac OS X using [Homebrew](http://brew.sh/), run `brew install sbt`. On Ubuntu run `sudo apt-get install sbt`.

**Option 2 - Obtain packaged binaries**

Once the first version of LOCO will have been published, the binaries will be available on the [releases](https://github.com/christinaheinze/loco-lib/releases) page. 

# Examples

## Ridge Regression

To run ridge regression locally on the 'climate' regression data set provided in the `data` directory, run:
{% highlight bash %}
spark-1.3.0/bin/spark-submit \
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
	
{% endhighlight %}

The estimated coefficients can be plotted as follows as each feature corresponds to one grid point on the globe. For more information on the data set, see [LOCO: Distributing Ridge Regression with Random Projections](http://arxiv.org/abs/1406.3469).

![regression_coefficients]({{ site.baseurl }}/images/beta_loco.png)

## SVM

To train a binary SVM with hinge loss locally on the 'dogs vs. cats' classification data set provided in the `data` directory, run:
{% highlight bash %}
spark-1.3.0/bin/spark-submit \
--class "LOCO.driver" \
--master local[4] \
--driver-memory 1G \
--executor-memory 1G \
--conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
--conf "spark.kryoserializer.buffer.max.mb=512" \
--conf "spark.executor.extraJavaOptions=-XX:+UseG1GC" \
--conf "spark.driver.maxResultSize=512m" \
target/scala-2.10/LOCO-assembly-csc-unserialized-0.1.jar \
--classification=true \
--optimizer=SDCA \
--numIterations=5000 \
--dataFormat=text \
--textDataFormat=spaces \
--separateTrainTestFiles=false \
--dataFile="data/dogs_vs_cats_n5000.txt" \
--center=false \
--centerFeaturesOnly=true \
--Proj=sparse \
--concatenate=false \
--CVKind=global \
--lambda=0.2 \
--nFeatsProj=200 \
--nPartitions=4 \
--nExecutors=1
{% endhighlight %}


# LOCO<sup>lib</sup> options

The following list provides a description of all options that can be provided to LOCO. 

`outdir` Directory where to save the summary statistics and the estimated coefficients as text files

`saveToHDFS` True if output should be saved on HDFS

`nPartitions` Number of blocks to partition the design matrix 

`nExecutors` Number of executors used. This information will be used to determine the tree depth in [`treeReduce`](http://spark.apache.org/docs/1.3.1/api/scala/index.html#org.apache.spark.rdd.RDD) when the random projections are added. A binary tree structure is used to minimise the memory requirements. 

`dataFormat` Can be either "text" or "object"

`textDataFormat` If `dataFormat` is "text", it can have the following formats: 

* "libsvm" : [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) format, e.g.:
{% highlight R %}
0 1:1.2 2:3.1 ...
1 1:-2.1 2:4.3 ...
{% endhighlight  %}

* "comma" : The response is separated by a comma from the features. The features are separated by spaces, e.g.:

{% highlight R %}
0, 1.2 3.1 ...
1, -2.1 4.3 ...
{% endhighlight  %}


* "spaces" : Both the response and the features are separated by spaces, e.g.:

{% highlight R %}
0 1.2 3.1 ...
1 -2.1 4.3 ...
{% endhighlight  %}

`dataFile` Path to the input data file

 `separateTrainTestFiles` True if training and test set are provided in different files
 
`trainingDatafile` If training and test set are provided in different files, path to the training data file

`testDatafile` If training and test set are provided in different files, path to the test data file

`proportionTest` If training and test set are _not_ provided separately, proportion of data set to use for testing

`myseed` Random seed

`classification` True if the problem at hand is a classification task, otherwise ridge regression will be performed

`lossClassification` Loss function to be used in classification (to be added)

`optimizer` Solver used in the local optimisation. Can be either ["SDCA"](#references) (stochastic dual coordinate ascent) or ["factorie"](#references). If the latter is chosen, ridge regression is optimised by L-BFGS.

`numIterations` If SDCA is chosen, number of iterations used

`checkDualityGap` If SDCA is chosen, true if duality gap should be computed after each iteration. Note that this is a very expensive operation as it requires a pass over the full local data sets (no communication required). Should only be used for tuning purposes.  

`stoppingDualityGap` If SDCA is chosen and `checkDualityGap` is set to true, duality gap at which optimisation should stop

`center` True if both the response and the features should be centred. Centering and scaling the data could also be done with the preprocessing package (see below)

`centerFeaturesOnly` True if only the features should be centred (e.g. for classification)

`projection` Random projection to use: can be either "sparse" or "SRHT"

`nFeatsProj` Projection dimension 

`concatenate` True is random projections should be concatenated, otherwise they are added

`CVKind` Can be either "global", "local", or "none"

* "global" performs the full LOCO algorithm for a provided sequence of regularisation parameters and returns the parameter value yielding the smallest misclassification rate in case of classification and the smallest MSE in case of regression. 

* "local" finds the optimal regularisation parameters for the local optimisation problems, i.e. the relevant training and test sets are splits of the local design matrices. 

`kfold` Number of splits to use for cross validation

`lambdaSeqFrom` Start of regularisation parameter sequence to use for cross validation

`lambdaSeqTo` End of regularisation parameter sequence to use for cross validation

`lambdaSeqBy` Step size for regularisation parameter sequence to use for cross validation

`lambda` If no cross validation should be performed (`CVKind=none`), regularisation parameter to use

## Choosing the projection dimension

The smallest possible projection dimension depends on the rank of the data matrix \\( \boldsymbol{X} \\). If you expect your data to be low-rank so that LOCO is suitable, we recommend using a projection dimension of about 10% of the number of features you are compressing. The latter depends on whether you choose to add or to concatenate the random features. This projection dimension should be used as a starting point, of course you can test whether your data set allows for a larger degree of compression by tuning the projection dimension together with the regularisation parameter \\( \lambda \\).
	
### Concatenating the random features
As described in the original LOCO paper, the first option for collecting the random projections from the other workers is to concatenate them and append these random features to the raw features. More specifically, each worker has \\( \tau = p / K \\) raw features which are compressed to \\( \tau\_{subs} \\) random features. These random features are then communicated and concatenating all random features from the remaining workers results in a dimensionality of the random features of \\( (K-1) \cdot \tau_{subs} \\). Finally, the full local design matrix, consisting of raw and random features, has dimension \\( n \times (\tau + (K-1) \cdot \tau\_{subs}) \\).

### Adding the random features
If the projection matrix is a random matrix, e.g. with entries in \\( \( 0, 1, -1\) \\) drawn with probabilities \\( \{ \frac{2}{3}, \frac{1}{6}, \frac{1}{6} \} \\), one can alternatively add the random projections. This is equivalent to projecting all raw features not belonging to worker \\( k \\) at once. If the data set is very low-rank, this scheme may allow for a smaller dimensionality of the random features than concatenation of the random features as we can now project from \\( (p - p/K)\\) to \\( \tau\_{subs} \\) instead of from \\( \tau = p/K \\) to \\( \tau\_{subs} \\).

# Preprocessing package
The preprocessing package 'preprocessingUtils' can be used to 

* center and/or scale the features and/or the response to have zero mean and unit variance, using [Spark MLlib](http://spark.apache.org/docs/1.2.1/mllib-guide.html)'s [`StandardScaler`](http://spark.apache.org/docs/1.2.1/mllib-feature-extraction.html#standardscaler).
* save data files in serialised format using the Kryo serialisation library. This code follows the example from @phatak-dev provided [here](http://blog.madhukaraphatak.com/kryo-disk-serialization-in-spark/).
* convert text files of the formats "libsvm", "comma", and "space" (see examples under [options](#locosuplibsup-options)) to object files with RDDs containing elements of type 
	- `DataPoint` (needed for LOCO, see details [below](#datapoint))
	- [`LabeledPoint`](http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.regression.LabeledPoint) (needed for the algorithms provided in Spark's machine learning library MLlib)
	- `Array[Double]` where the first entry is the response, followed by the features

## Data Structures
The preprocessing package defines two case classes LOCO relies on:

### DataPoint
The case class `DataPoint` that is modelled after MLlib's [`LabeledPoint`](http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.regression.LabeledPoint):
{% highlight Scala %}
case class DataPoint(label: Double, features: breeze.linalg.Vector[Double])
{% endhighlight %}
### FeatureVector
The case class `FeatureVector` contains all observations of a particular variable as a vector in the field `observations`. The field `index` serves as an identifier for the feature vector.
{% highlight Scala %}
case class FeatureVector(index : Int, observations: breeze.linalg.Vector[Double])
{% endhighlight %}

## Example

## preprocessingUtils options 

The following list provides a description of all options that can be provided to the package 'preprocessingUtils'. 

`outdir` Directory where to save the converted data files

`nPartitions` Number of partitions to use

`dataFormat` Can be either "text" or "object"

`textDataFormat` If `dataFormat` is "text", it can have the following formats: "libsvm", "comma", "spaces" (see [here](#locosuplibsup-options) for examples)

`dataFile` Path to the input data file

 `separateTrainTestFiles` True if training and test set are provided in different files
 
`trainingDatafile` If training and test set are provided in different files, path to the training data file

`testDatafile` If training and test set are provided in different files, path to the test data file

`proportionTest` If training and test set are _not_ provided separately, proportion of data set to use for testing

`myseed` Random seed

`outputTrainFileName` File name for file containing the training data

`outputTestFileName` File name for file containing the test data

`outputClass` Specifies the type of the elements in the output RDDs : can be `DataPoint`, `LabeledPoint` or `DoubleArray`

`twoOutputClasses` True if same training/test pair should be saved in two different formats

`secondOutputClass` If `twoOutputClasses` is true, specifies the type of the elements in the corresponding output RDDs 

`centerFeatures` True if features should be centred to have zero mean

`centerResponse` True if response should be centred to have zero mean 

`scaleFeatures` True if features should be scaled to have unit variance

`scaleResponse` True if response should be scaled to have unit variance

# Recommended Spark settings
Note that the benefit of some of these setting highly depends on the particular architecture you will be using, i.e. we cannot guarantee that they will yield optimal performance of LOCO.

* Use Kryo serialisation
 {% highlight bash %}
--conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" 
{% endhighlight %}

* Increase the maximum allowable size of the Kryo serialization buffer
{% highlight bash %}
--conf "spark.kryoserializer.buffer.max.mb=512" 
{% endhighlight %}


*  Use Java's more recent "garbage first" garbage collector which was designed for heaps larger than 4GB if there are no memory constraints 

{% highlight bash %}
--conf "spark.executor.extraJavaOptions=-XX:+UseG1GC"
{% endhighlight %}

* Set the total size of serialised results of all partitions large enough to allow for the random projections to be send to the driver
{% highlight bash %}
--conf "spark.driver.maxResultSize=3g" 
{% endhighlight %}


# References
The LOCO algorithm is described in the following paper:

 * _Heinze, C., McWilliams, B., Meinshausen, N., Krummenacher, G., Vanchinathan, H. P. (2014) [LOCO: Distributing Ridge Regression with Random Projections](http://arxiv.org/abs/1406.3469)_

 Further references:
 
 * _McCallum, A., Schultz, K., Singh, S. [FACTORIE: Probabilistic Programming via Imperatively Defined Factor Graphs](http://people.cs.umass.edu/~mccallum/papers/factorie-nips09.pdf). Neural Information Processing Systems (NIPS), 2009._
 * _Shalev-Shwartz, S. and Zhang, T. [Stochastic dual coordinate ascent methods for regularized loss minimization](http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf). JMLR, 14:567–599, February 2013c._
 