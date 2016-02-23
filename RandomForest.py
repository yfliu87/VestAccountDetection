
from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.tree import RandomForest

def trainModel(trainingData):
	model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, 
											numTrees=3, featureSubsetStrategy="auto", impurity='gini',
											maxDepth=5, maxBins=32)

	print '\nTraining RandomForest model finished'
	return model


def trainOptimalModel(trainingData):
	return None


def evaluateModel(model, testData):
	predictions = model.predict(testData.map(lambda item: item.features))
	labelsAndPredictions = testData.map(lambda item: item.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda (v,p): v != p).count()/float(testData.count())

	print "\RandomForest Model Evaluation"
	print "\nTest Error = " + str(testErr)