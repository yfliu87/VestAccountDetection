#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
import ClassificationMain as classification
import ClusterMain as cluster
import ComputeModule as compute
import PredefinedValues as pv
import FileParser as fp
import Utils

sc = SparkContext()
clusterAccountMap = {}

def demo(count):
	Utils.logMessage("\nJob started %s rounds" %str(count))
	Utils.logMessage("\nInitial accounts %s " %str(pv.truncateLineCount - 1))

	for idx in xrange(pv.truncateLineCount, pv.truncateLineCount + count):
		clusterModel, classificationModel = loadModel()

		#take random account from merged file
		df = fp.readData(pv.mergedAccountFile)
		record = df.loc[idx]

		#organize feature count
		compressedRecord = countByFeatures(record)
		labeledRecord = LabeledPoint(compressedRecord[0], compressedRecord[2:])

		#predict cluster
		predictedLabel = classificationModel.predict(labeledRecord.features)

		if predictedLabel <= 1:
			print "\nPredicted label: %s, safe account, go for next" %str(predictedLabel)
			continue

		else:
			print "\nPredicted label: %s, risky account, double check using cluster model" %str(predictedLabel)

			#calculate similarity with existing simMatrix
			sim = calculateSim(df, idx)
			Utils.logMessage("\ncalculateSim done")

			mostSimilarAccountIdx = sim.index(max(sim))
			newLabel = getLabelByIdx(df, mostSimilarAccountIdx)

			Utils.logMessage("\nLabel of most similar account is %s" %str(newLabel))

			if newLabel >= predictedLabel:
				print "\nSuspecious account, mark as training data for next round"
				pv.truncateLineCount += idx

				df.loc[idx] = refreshRecord(record, predictedLabel)

				df.to_csv(pv.mergedAccountFile, index=False, encoding='utf-8')

				#retrain classification model
				classification.run(sc)

				#retrain cluster model
				cluster.run(sc)
			else:
				print "\nLow risk account, go for next"

	Utils.logMessage("Job Finished!")


def loadModel():
	clusterModel = KMeansModel.load(sc, pv.clusterModelPath)
	buildClusterAccountMap(pv.clusterIDCenterFile)
	classificationModel = DecisionTreeModel.load(sc, pv.classificationModelPath)
	Utils.logMessage("\nLoad cluster & classification model finished")
	return clusterModel, classificationModel

def buildClusterAccountMap(clusterAccountFilePath):
	df = fp.readData(clusterAccountFilePath)
	for idx in xrange(df.shape[0]):
		pin = df.loc[idx]['buyer_pin']
		cluster = df.loc[idx]['center']

		if cluster not in clusterAccountMap:
			clusterAccountMap[cluster] = []

		clusterAccountMap[cluster].append(idx)

def countByFeatures(record):
	ipNum = len(record['buyer_ip'].split('|'))
	devIDNum = len(record['equipment_id'].split('|'))
	addrNum = len(record['buyer_poi'].split('|'))
	promotionNum = len(record['promotion_id'].split('|'))
 
	return (record['label'], record['buyer_pin'], ipNum ** 2, devIDNum ** 3, addrNum ** 2, promotionNum)


def calculateSim(df, matSize):
	cur = df.loc[matSize]
	ret = []
	for i in xrange(matSize):
		ret.append(compute.computeSimilarity(df.loc[i], cur))
	return ret


def checkSimilarAccounts(df, curAccount, clusteredAccounts):
	maxSim = 0.0
	candidateIdx = -1
	for item in clusteredAccounts:
		account = df.loc(item)
		sim = compute.computeSimilarity(curAccount, account)
		if sim > maxSim:
			maxSim = sim
			candidateIdx = item

	return getLabelByIdx(df, candidateIdx)


def getLabelByIdx(df,idx):
	return df.loc[idx]['label']

def refreshRecord(record, newLabel):
	retList = record.tolist()
	labelIdx = retList.index(record['label'])
	retList[labelIdx] = newLabel
	return np.array(retList)


def run():
	#preprocess rule output file, mark, combine
	classification.preprocess()

	Utils.logMessage("\nPretraining model started")

	#train cluster model
	cluster.run(sc)

	#train classification model
	classification.run(sc)

	Utils.logMessage("\nPretraining model finished")

	#random pick 10 records and make prediction
	demo(10)

if __name__ == '__main__':
	run()
