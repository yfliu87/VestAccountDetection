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
import codecs
import FileParser as fp
import Evaluation as eva
import PredefinedValues as pv
import Utils

#sparkContext = SparkContext()
#writer = open('/home/yifei/TestData/data/realdata/classification_50000.csv','w')
optimalModel = None
minTrainingError = 1.0
minTestError = 1.0


def preprocess():
	
	rawDataFrame = pd.read_table(pv.confirmedAccountFile, sep=',',encoding='utf-8')
	rawDataFrame['label'] = [3 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.confirmedAccountFile, header=False, index=False,encoding='utf-8')

	rawDataFrame = pd.read_table(pv.rule1AccountFile, sep=',',encoding='utf-8')
	rawDataFrame['label'] = [2 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.rule1AccountFile, header=False, index=False,encoding='utf-8')
	'''
	rawDataFrame = pd.read_table(pv.rule2AccountFile, sep=',',encoding='utf-8')
	rawDataFrame['label'] = [2 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.rule2AccountFile, header=False, index=False,encoding='utf-8')
	'''
	
	rawDataFrame = pd.read_table(pv.rule3AccountFile, sep=',',encoding='utf-8')
	rawDataFrame['label'] = [2 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.rule3AccountFile, header=False, index=False,encoding='utf-8')

	rawDataFrame = pd.read_table(pv.rule4AccountFile, sep=',',encoding='utf-8')
	rawDataFrame['label'] = [1 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.rule4AccountFile, header=False, index=False,encoding='utf-8')

	rawDataFrame = pd.read_table(pv.rule5AccountFile, sep=',',encoding='utf-8')
	rawDataFrame['label'] = [1 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.rule5AccountFile, header=False, index=False,encoding='utf-8')

	rawDataFrame = pd.read_table(pv.rule6AccountFile, sep=',',encoding='utf-8')
	rawDataFrame['label'] = [0 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.rule6AccountFile, header=False, index=False,encoding='utf-8')

	rawDataFrame = pd.read_table(pv.randomAccountFile, sep=',',encoding='utf-8')
	rawDataFrame['label'] = [0 for i in xrange(rawDataFrame.shape[0])]
	rawDataFrame.to_csv(pv.randomAccountFile, header=False, index=False,encoding='utf-8')

	Utils.logMessage("\nMark account finished")
	pv.fileColumns.append('label')
	merge(pv.mergedAccountFile, [pv.confirmedAccountFile, pv.randomAccountFile, pv.rule1AccountFile, pv.rule2AccountFile, pv.rule3AccountFile, pv.rule4AccountFile, pv.rule5AccountFile, pv.rule6AccountFile])

def merge(targetFile, files):
	fWriter = codecs.open(targetFile, 'w','utf-8')
	fWriter.write(','.join(pv.fileColumns))
	fWriter.write('\n')

	for f in files:
		fReader = codecs.open(f, 'r','utf-8')

		line = fReader.readline()
		while line:
			fWriter.write(line)
			line = fReader.readline()

		fReader.close()

	fWriter.close()

	Utils.logMessage("\nMerge account finished")

def run(sparkContext):
	#global sparkContext, writer
	Utils.logMessage("\nClassification model started")
	pd.read_table(pv.processedFile, sep=',',encoding='utf-8').to_csv(pv.processedFile, header=False, index=False,encoding='utf-8')
	truncatedAccounts = sparkContext.textFile(pv.processedFile).take(pv.truncateLineCount - 1)
	rawData = sparkContext.parallelize(truncatedAccounts).map(countByFeatures).map(lambda item: LabeledPoint(item[0], Vectors.dense(item[2:])))

	for ratio in [0.7]:
		for impurity in ['entropy']:
			for maxDepth in [4]:
				for maxBin in [16]:
					runWithParam(sparkContext, rawData, ratio, impurity, maxDepth, maxBin)


def runWithParam(sparkContext, rawData, ratio, impurity, maxDepth, maxBin):
	trainingSet, testSet = rawData.randomSplit([ratio, 1-ratio])

	decisionTreeModel, trainingError, testError = DecisionTreeProcess(trainingSet, testSet, impurity, maxDepth, maxBin)
	print '\nDecision Tree Training Err: %s, Test Err: %s' %(str(trainingError), str(testError))
	decisionTreeModel.save(sparkContext, pv.classificationModelPath)

	'''
	writer.write('\nCurrent run ratio %s, maxDepth %s, maxBin %s, impurity %s' %(str(ratio), str(maxDepth), str(maxBin), impurity))
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

	#binary classification
	SVMModel, trainingError, testError = SVMProcess(trainingSet, testSet)
	print '\nSVM Training Err: %s, Test Err: %s' %(str(trainingError), str(testError))

	#writer.write('\n\tSVM TrainingError %s, TestError %s' %(str(trainingError), str(testError)))
	#recordOptimal(trainingError, testError, SVMModel)
	'''

def countByFeatures(item):
	items = item.split(',')
	ipNum = len(items[1].split('|'))
	devIDNum = len(items[2].split('|'))
	addrNum = len(items[3].split('|'))
	promotionNum = len(items[4].split('|'))
 
	return (items[-1], items[0], ipNum ** 2, devIDNum ** 3, addrNum ** 2, promotionNum)


def recordOptimal(trainingError,testError, model):
	global minTrainingError, minTestError, optimalModel
	if trainingError < minTrainingError and testError < minTestError:
		minTrainingError = trainingError
		minTestError = testError
		optimalModel = model 


def DecisionTreeProcess(trainingSet, testSet, imp, dtMaxDepth, dtMaxBins):
	
	decisionTreeModel = DecisionTree.trainClassifier(trainingSet, numClasses = 4,categoricalFeaturesInfo={},
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
	pass
	#global writer, optimalModel, minTrainingError, minTestError

	#preprocess()

	#run()

	#writer.write('\n\nFinal optimal param\nminTrainingError %s, minTestError %s' %(str(minTrainingError), str(minTestError)))
	#writer.close()
	#optimalModel.save(sparkContext,'/home/yifei/TestData/data/realdata/classification.model')
