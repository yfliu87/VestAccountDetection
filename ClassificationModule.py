from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
import pandas as pd
import Utils

def process(sc, eigenVecFile, markedClusterFile):
	filteredEigenVec = sc.textFile(eigenVecFile).map(lambda item: removeVirtualPart(item)).collect()
	clusterIDs = sc.textFile(markedClusterFile).map(lambda item: extractClusterID(item)).collect()
	clusterIdEigenVecMapRDD = sc.parallelize(clusterIDs).zip(sc.parallelize(filteredEigenVec))
	labeledClusterIdEigenVecMapRdd = clusterIdEigenVecMapRDD.map(lambda item: LabeledPoint(item[0], item[1]))

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
