#-*-coding:utf-8-*-
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel 
from pyspark.mllib.tree import RandomForest, RandomForestModel 
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
import pandas as pd
import FileParser as fp
import Evaluation as eva
import PredefinedValues as pv

sparkContext = SparkContext()
writer = open('/home/yifei/TestData/data/realdata/classification_50000.csv','w')
optimalModel = None
minTrainingError = 1.0
minTestError = 1.0


def preprocess():
	rawDataFrame = pd.read_table(pv.confirmedAccountFile, sep=',')
	rawDataFrame['label'] = [1 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.confirmedAccountFile, header=False, index=False)

	rawDataFrame = pd.read_table(pv.randomAccountFile, sep=',')
	rawDataFrame['label'] = [0 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.randomAccountFile, header=False, index=False)

	merge(pv.mergedAccountFile, [pv.confirmedAccountFile, pv.randomAccountFile])

def merge(targetFile, files):
	fWriter = open(targetFile, 'w')

	for f in files:
		fReader = open(f, 'r')

		line = fReader.readline()
		while line:
			fWriter.write(line)
			line = fReader.readline()

		fReader.close()

	fWriter.close()


def run():
	global sparkContext, writer

	rawData = sparkContext.textFile(pv.mergedAccountFile).map(countByFeatures).map(lambda item: LabeledPoint(item[0], Vectors.dense(item[2:])))

	for ratio in [0.7, 0.8]:
		for impurity in ['entropy','gini']:
			for maxDepth in [5,6,7,8]:
				for maxBin in [16,32]:
					runWithParam(rawData, ratio, impurity, maxDepth, maxBin)


def runWithParam(rawData, ratio, impurity, maxDepth, maxBin):
	trainingSet, testSet = rawData.randomSplit([ratio, 1-ratio])

	decisionTreeModel, trainingError, testError = DecisionTreeProcess(trainingSet, testSet, impurity, maxDepth, maxBin)
	print '\nDecision Tree Training Err: %s, Test Err: %s' %(str(trainingError), str(testError))

	writer.write('\nCurrent run ratio %s, maxDepth %s, maxBin %s' %(str(ratio), str(maxDepth), str(maxBin)))
	writer.write('\n\tDecision Tree TrainingError %s, TestError %s' %(str(trainingError), str(testError)))
	recordOptimal(trainingError, testError, decisionTreeModel)

	randomForestModel, trainingError, testError = RandomForestProcess(trainingSet, testSet, impurity, maxDepth, maxBin)
	print '\nRandom Forest Training Err: %s, Test Err: %s' %(str(trainingError), str(testError))

	naiveBayesModel, trainingError, testError = NaiveBayesProcess(trainingSet, testSet)
	print '\nNaive Bayes Training Err: %s, Test Err: %s' %(str(trainingError), str(testError))

	#writer.write('\n\tNaive Bayes TrainingError %s, TestError %s' %(str(trainingError), str(testError)))
	#recordOptimal(trainingError, testError, decisionTreeModel)

	logRegressionModel, trainingError, testError = LogisticRegressionProcess(trainingSet, testSet)
	print '\nLogistic Regression Training Err: %s, Test Err: %s' %(str(trainingError), str(testError))

	#writer.write('\n\tLogistic Regression TrainingError %s, TestError %s' %(str(trainingError), str(testError)))
	#recordOptimal(trainingError, testError, logRegressionModel)

	SVMModel, trainingError, testError = SVMProcess(trainingSet, testSet)
	print '\nSVM Training Err: %s, Test Err: %s' %(str(trainingError), str(testError))

	#writer.write('\n\tSVM TrainingError %s, TestError %s' %(str(trainingError), str(testError)))
	#recordOptimal(trainingError, testError, SVMModel)


def countByFeatures(item):
	items = item.split(',')
	devIDNum = len(items[1].split('|'))
	ipNum = len(items[2].split('|'))
	addrNum = len(items[3].split('|'))
	promotionNum = len(items[4].split('|'))

	return (items[-1], items[0], ipNum, devIDNum, addrNum, promotionNum)


def recordOptimal(trainingError,testError, model):
	global minTrainingError, minTestError, optimalModel
	if trainingError < minTrainingError and testError < minTestError:
		minTrainingError = trainingError
		minTestError = testError
		optimalModel = model 


def DecisionTreeProcess(trainingSet, testSet, imp, dtMaxDepth, dtMaxBins):
	
	decisionTreeModel = DecisionTree.trainClassifier(trainingSet, numClasses = 2,categoricalFeaturesInfo={},
														impurity=imp,maxDepth=dtMaxDepth, maxBins=dtMaxBins)


	predictions = decisionTreeModel.predict(trainingSet.map(lambda item: item.features))
	trainingLabelsAndPredictions = trainingSet.map(lambda item: item.label).zip(predictions)
	trainingError = eva.calculateErrorRate(trainingLabelsAndPredictions)

	predictions = decisionTreeModel.predict(testSet.map(lambda item: item.features))
	testLabelsAndPredictions = testSet.map(lambda item: item.label).zip(predictions)
	testError = eva.calculateErrorRate(testLabelsAndPredictions)

	return decisionTreeModel, trainingError, testError

def RandomForestProcess(trainingSet, testSet, imp, dtMaxDepth, dtMaxBins):
	randomForestModel = RandomForest.trainClassifier(trainingSet, numClasses = 2, categoricalFeaturesInfo={},
														numTrees = 3, featureSubsetStrategy="auto",
														impurity=imp,maxDepth=dtMaxDepth, maxBins=dtMaxBins)

	predictions = randomForestModel.predict(trainingSet.map(lambda item: item.features))
	trainingLabelsAndPredictions = trainingSet.map(lambda item: item.label).zip(predictions)
	trainingError = eva.calculateErrorRate(trainingLabelsAndPredictions)

	predictions = randomForestModel.predict(testSet.map(lambda item: item.features))
	testLabelsAndPredictions = testSet.map(lambda item: item.label).zip(predictions)
	testError = eva.calculateErrorRate(testLabelsAndPredictions)

	return randomForestModel, trainingError, testError

def NaiveBayesProcess(trainingSet, testSet):
	naiveBayesModel = NaiveBayes.train(trainingSet)

	predictions = naiveBayesModel.predict(trainingSet.map(lambda item: item.features))
	trainingLabelsAndPredictions = trainingSet.map(lambda item: item.label).zip(predictions)
	trainingError = eva.calculateErrorRate(trainingLabelsAndPredictions)

	predictions = naiveBayesModel.predict(testSet.map(lambda item: item.features))
	testLabelsAndPredictions = testSet.map(lambda item: item.label).zip(predictions)
	testError = eva.calculateErrorRate(testLabelsAndPredictions)

	return naiveBayesModel, trainingError, testError


def LogisticRegressionProcess(trainingSet, testSet):
	logRegressionModel = LogisticRegressionWithLBFGS.train(trainingSet)

	predictions = logRegressionModel.predict(trainingSet.map(lambda item: item.features))
	trainingLabelsAndPredictions = trainingSet.map(lambda item: item.label).zip(predictions)
	trainingError = eva.calculateErrorRate(trainingLabelsAndPredictions)

	predictions = logRegressionModel.predict(testSet.map(lambda item: item.features))
	testLabelsAndPredictions = testSet.map(lambda item: item.label).zip(predictions)
	testError = eva.calculateErrorRate(testLabelsAndPredictions)

	return logRegressionModel, trainingError, testError


def SVMProcess(trainingSet, testSet):
	
	SVMModel = SVMWithSGD.train(trainingSet, iterations=100)

	predictions = SVMModel.predict(trainingSet.map(lambda item: item.features))
	trainingLabelsAndPredictions = trainingSet.map(lambda item: item.label).zip(predictions)
	trainingError = eva.calculateErrorRate(trainingLabelsAndPredictions)

	predictions = SVMModel.predict(testSet.map(lambda item: item.features))
	testLabelsAndPredictions = testSet.map(lambda item: item.label).zip(predictions)
	testError = eva.calculateErrorRate(testLabelsAndPredictions)

	return SVMModel, trainingError, testError



if __name__ == '__main__':
	global sparkContext, writer, optimalModel, minTrainingError, minTestError

	#preprocess()

	run()

	writer.write('\n\nFinal optimal param\nminTrainingError %s, minTestError %s' %(str(minTrainingError), str(minTestError)))
	writer.close()
	optimalModel.save(sparkContext,'/home/yifei/TestData/data/realdata/classification_50000.model')
