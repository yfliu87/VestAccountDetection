
from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.tree import DecisionTree


def trainModel(trainingData):
	model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5,maxBins=32)
	print '\nTraining DecisionTree model finished'
	return model


def trainOptimalModel(trainingData):
	return None
	

def evaluateModel(model, testData):
	predictions = model.predict(testData.map(lambda item: item.features))
	labelsAndPredictions = testData.map(lambda item: item.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda (v,p): v != p).count()/float(testData.count())

	print "\nDecisionTree Model Evaluation"
	print "\nTest Error = " + str(testErr)

