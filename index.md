---
layout: dis_template
---

NOTE: THIS PAGE IS STILL UNDER DEVELOPMENT AND THE SOFTWARE HAS NOT BEEN PUBLISHED

# Problem setting

LOCO is a LOw-COmmunication distributed algorithm for \\( \ell_2 \\) - penalised convex estimation problems of the form

\\[ \min_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = \frac{1}{n} \sum\_{i=1}^n f_i(\boldsymbol{\beta}^\top \mathbf{x}_i) + \frac{\lambda}{2} \Vert \boldsymbol{\beta} \Vert^2_2 \\]

LOCO is suitable in a setting where the number of features \\( p \\) is very large so that splitting the design matrix 
across _features_ rather than observations is reasonable. For instance, the number of observations \\( n \\) can be smaller than \\( p \\) or on the same order. LOCO is expected to yield good performance when the rank of the data matrix \\( \boldsymbol{X} \\) is much lower than the actual dimensionality of the observations \\( p \\).

<h1> Table of Contents </h1>
* auto-gen TOC:
{:toc}

# One-shot communication scheme

One large advantage of LOCO over iterative distributed algorithms such as distributed stochastic gradient descent (SGD) is that LOCO only needs one round of communication before the results are send back to the driver. Therefore, the communication cost -- which is generally the bottleneck in distributed computing -- is very low.

LOCO proceeds as follows. As a preprocessing step, the features need to be distributed across processing units by randomly partitioning the data into K blocks. After this first step, some number of feature vectors are stored on each worker. We shall call these features the "raw" features of worker \\(k\\). Subsequently, each worker applies a dimensionality reduction on its raw features by using a random projection. The resulting features are called the "random features" of worker \\(k\\). Then each worker sends its random features to the other workers. Each worker then adds or concatenates the random features from the other workers and appends these random features to its own raw features. Using this scheme, each worker has access to its raw features and, additionally, to a compressed version of the remaining workers' raw features. Using this design matrix, each worker estimates coefficients locally and returns the ones for its own raw features. As these were learned using the contribution from the random features, they approximate the optimal coefficients sufficiently well. The final estimate returned by LOCO is simply the concatenation of the raw feature coefficients returned from the \\( K \\) workers. 

LOCO's distribution scheme is illustrated in the following figure.

![LOCO]({{ site.baseurl }}/images/loco2.png)

# Obtaining the software

**Option 1 - Build with sbt**

Checkout the project repository

	git clone https://github.com/christinaheinze/loco-lib.git

and build the package with
{% highlight bash %}
cd loco-lib
sbt assembly
{% endhighlight %}

To install `sbt` on Mac OS X using [Homebrew](http://brew.sh/), run `brew install sbt`. On Ubuntu run `sudo apt-get install sbt`.

**Option 2 - Obtain packaged binaries**

Once the first version of LOCO will have been published, the binaries will be available on the [releases](https://github.com/christinaheinze/loco-lib/releases) page. 

# Examples

## Ridge Regression

To run ridge regression locally on the regression data set provided in the `data` directory, run:
{% highlight bash %}
spark-1.2.0/bin/spark-submit \
	--class "LOCO.driver" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/
{% endhighlight %}

## SVM

To train a binary SVM with hinge loss locally on the classification data set provided in the `data` directory, run:
{% highlight bash %}
spark-1.2.0/bin/spark-submit \
	--class "LOCO.driver" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/
{% endhighlight %}


# Options

The following list provides a description of all options that can be provided to LOCO. 

`outdir : String` Directory where to save the summary statistics and the estimated coefficients as text files

`saveToHDFS : Boolean` True if output should be saved on HDFS

`nPartitions : Int` Number of blocks to partition the design matrix 

`dataFormat : String` Can be either "text" or "object"

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

`classification` True if the problem at hand is a classification task, otherwise regression will be performed

`lossClassification` Loss function to be used in classification (to be added)

`optimizer` Solver used in the local optimisation. Can be either "SDCA" (stochastic dual coordinate ascent) or "factorie". If the latter is chosen, ridge regression is optimised by L-BFGS.

`numIterations` If SDCA is chosen, number of iterations used

`checkDualityGap` If SDCA is chosen, true if duality gap should be computed after each iteration. Note that this is a very expensive operation as it requires a pass over the full local data sets (no communication required). Should only be used for tuning purposes.  

`stoppingDualityGap` If SDCA is chosen and `checkDualityGap` is set to true, duality gap at which optimisation should stop

`center` True if both the response and the features should be centred. Centering and scaling the data could also be done with the preprocessing package (see below)

`centerFeaturesOnly` True if only the features should be centred (e.g. for classification)

`projection` Random projection to use: can be either "sparse" or "SRHT"

`nFeatsProj` Projection dimension 

`concatenate` True is random projections should be concatenated, otherwise they are added

`CVKind` Can be either "global", "local", or "none"

* "Global" performs the full LOCO algorithm for a provided sequence of regularisation parameters and returns the parameter value yielding the smallest misclassification rate in case of classification and the smallest MSE in case of regression. 

* "Local" finds the optimal regularisation parameters for the local optimisation problems, i.e. the relevant training and test sets are splits of the local design matrices. 

`kfold` Number of splits to use for cross validation

`lambdaSeqFrom` Start of regularisation parameter sequence to use for cross validation

`lambdaSeqTo` End of regularisation parameter sequence to use for cross validation

`lambdaSeqBy` Step size for regularisation parameter sequence to use for cross validation

`lambda` If no cross validation should be performed, regularisation parameter to use

## Choosing the projection dimension

The smallest possible projection dimension depends on the rank of the data matrix \\( \boldsymbol{X} \\). If you expect your data to be low-rank so that LOCO is suitable, we recommend using a projection dimension of about 10% of the number of features you are compressing. The latter depends on whether you choose to add or to concatenate the random features.
	
### Adding the random features

### Concatenating the random features

# Preprocessing package
The preprocessing package can be used to 

* center and/or scale the features and/or the response to have zero mean and unit variance
* convert text files of various formats to the case classes `DataPoint` (needed for LOCO) or `LabeledPoint` (needed for the Spark machine learning library MLlib)
* save the data files in serialised format using the Kryo serialisation library

## Example

## Options 

# Recommended Spark settings
* Use Kryo serialisation
 {% highlight bash %}
--conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" 
{% endhighlight %}

* Increase the maximum allowable size of the Kryo serialization buffer
{% highlight bash %}
--conf "spark.kryoserializer.buffer.max.mb=512" 
{% endhighlight %}


*  Use Java's more recent "garbage first" garbage collector which was designed for heaps larger than 4GB 

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