from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
import pandas as pd
import PredefinedValues as preVal
import Utils
import Evaluation as eva

def process(sc, eigenVecFile, markedClusterFile):
	filteredEigenVec = sc.textFile(eigenVecFile).map(lambda item: removeVirtualPart(item)).collect()
	clusterIDs = sc.textFile(markedClusterFile).map(lambda item: extractClusterID(item)).collect()
	clusterIdEigenVecMapRDD = sc.parallelize(clusterIDs).zip(sc.parallelize(filteredEigenVec))
	labeledClusterIdEigenVecMapRdd = clusterIdEigenVecMapRDD.map(lambda item: LabeledPoint(item[0], item[1]))

	trainingSet, testSet = labeledClusterIdEigenVecMapRdd.randomSplit([7,3])

	decisionTreeModel = DecisionTree.trainClassifier(trainingSet, numClasses = preVal.clusterNum,
														categoricalFeaturesInfo={},impurity='gini',maxDepth=3, maxBins=8)

	predictions = decisionTreeModel.predict(trainingSet.map(lambda item: item.features))
	trainingLabelsAndPredictions = trainingSet.map(lambda item: item.label).zip(predictions)
	eva.calculateErrorRate(trainingLabelsAndPredictions)

	predictions = decisionTreeModel.predict(testSet.map(lambda item: item.features))
	testLabelsAndPredictions = testSet.map(lambda item: item.label).zip(predictions)
	eva.calculateErrorRate(testLabelsAndPredictions)


def removeVirtualPart(item):
	return [extractRealPart(sub) for sub in item.split(',')]


def extractRealPart(item):
	left = item.find('(')
	right = item.find('+')
	if left != -1 and right != -1:
		return float(item[left+1:right])
	else:
		return 0.0


def extractClusterID(item):
	return int(item[-1:])
