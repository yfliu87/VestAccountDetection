import RandomForest
import DecisionTree
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.util import MLUtils

def run(searchForOptimal, basepath, filepath):
	sc = buildContext()

	trainingData, testData = loadData(sc, basepath, filepath)

	if searchForOptimal:
		#optimalRandomForestModel = RandomForest.trainOptimalModel(trainingData)
		optimalDecisionTreeModel = DecisionTree.trainOptimalModel(trainingData, testData)
	else:
		#randomForestModel = RandomForest.trainModel(trainingData)
		#RandomForest.evaluateModel(randomForestModel, testData)
		
		decisionTreeModel = DecisionTree.trainModel(trainingData)
		DecisionTree.evaluateModel(decisionTreeModel, testData)


def buildContext():
	conf = SparkConf().setAppName('ClassificationModel')
	print "\nBuild context finished"
	return SparkContext(conf = conf)


def loadData(sc, basepath, filepath):
	data = MLUtils.loadLibSVMFile(sc, os.path.join(basepath, filepath))
	trainingData, testData = data.randomSplit([0.7,0.3])
	print '\nLoad data finished'
	return trainingData, testData


if __name__ == '__main__':
	basepath = '/home/yifei/TestData/data'
	filepath = 'a9a_data.txt'
	searchForOptimal = True 
	run(searchForOptimal, basepath, filepath)

