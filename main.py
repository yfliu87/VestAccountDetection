import RandomForest
import DecisionTree
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.util import MLUtils

def run(basepath, filepath):
	sc = buildContext()

	trainingData, testData = loadData(sc, basepath, filepath)

	randomForestModel = RandomForest.trainModel(trainingData)
	RandomForest.evaluateModel(randomForestModel, testData)

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

'''
def modelEvaluation(model, testData):
	predictions = model.predict(testData.map(lambda x: x.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda (v,p): v != p).count()/float(testData.count())

	print "\nTest Error = " + str(testErr)
	#print "\nLearned classification model:"
	#print model.toDebugString()
'''

if __name__ == '__main__':
	basepath = '/home/yifei/TestData/data'
	filepath = 'a9a_data.txt'
	run(basepath, filepath)

