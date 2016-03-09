from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext
import pandas as pd
import Utils

def process(sc, eigenVecFile, markedClusterFile):
	eigenVecRDD = sc.textFile(eigenVecFile)
	#print eigenVecRDD.take(3)
	filteredEigenVecRDD = eigenVecRDD.map(lambda item: removeVirtualPart(item))
	#print filteredEigenVecRDD.take(3)


def removeVirtualPart(item):
	return [extract(sub) for sub in item.split(',')]


def extract(item):
	left = item.find('(')
	right = item.find('+')
	if left != -1 and right != -1:
		return float(item[left+1:right])
	else:
		return 0.0