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

def train(sparkContext):
	Utils.logMessage("\nClassification model started")
	pd.read_table(pv.processedFile, sep=',',encoding='utf-8').to_csv(pv.processedFile, header=False, index=False,encoding='utf-8')
	truncatedAccounts = sparkContext.textFile(pv.processedFile).take(pv.truncateLineCount - 1)
	rawData = sparkContext.parallelize(truncatedAccounts).map(countByFeatures).map(lambda item: LabeledPoint(item[0], Vectors.dense(item[2:])))

	trainWithParam(sparkContext, rawData, 0.7, 'entropy', 4, 16)


def trainWithParam(sparkContext, rawData, ratio, impurity, maxDepth, maxBin):
	trainingSet, testSet = rawData.randomSplit([ratio, 1-ratio])

	decisionTreeModel, trainingError, testError = DecisionTreeProcess(trainingSet, testSet, impurity, maxDepth, maxBin)
	print '\nDecision Tree Training Err: %s, Test Err: %s' %(str(trainingError), str(testError))
	decisionTreeModel.save(sparkContext, pv.classificationModelPath)


def countByFeatures(item):
	items = item.split(',')
	ipNum = len(items[1].split('|'))
	devIDNum = len(items[2].split('|'))
	addrNum = len(items[3].split('|'))
	promotionNum = len(items[4].split('|'))
 
	return (items[-1], items[0], ipNum ** 2, devIDNum ** 3, addrNum ** 2, promotionNum)


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


if __name__ == '__main__':
	pass
