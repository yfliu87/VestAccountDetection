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

def demo(count):
	for idx in xrange(pv.truncateLineCount, pv.truncateLineCount + count):
		if pv.outputDebugMsg:
			Utils.logMessage("\nUser %s" %str(idx))

		clusterModel, classificationModel = loadModel()

		df = fp.readData(pv.mergedAccountFile)
		record = df.loc[idx]

		compressedRecord = countByFeatures(record)

		labeledRecord = LabeledPoint(compressedRecord[0], compressedRecord[2:])

		predictedLabel = classificationModel.predict(labeledRecord.features)

		if predictedLabel <= 1:
			Utils.logMessage("\nPredicted label: %s, safe account, go for next" %str(predictedLabel))
			continue
		else:
			Utils.logMessage("\nPredicted label: %s, risky account, double check using cluster model" %str(predictedLabel))

			#calculate similarity with existing simMatrix
			sim = calculateSim(df, idx)

			if max(sim) > simThreshold:
				mostSimilarAccountIdx = sim.index(max(sim))
				newLabel = getLabelByIdx(df, mostSimilarAccountIdx)

				Utils.logMessage("\n\tLabel of most similar account is %s" %str(newLabel))

				if newLabel >= predictedLabel:
					updateLabelForNextRoundTrain()
				else:
					Utils.logMessage("\n\t\tLow risk account, go for next")
			else:
				updateLabelForNextRoundTrain(idx, df, record, predictedLabel, sc)

	Utils.logMessage("Job Finished!")


def updateLabelForNextRoundTrain(idx, df, record, predictedLabel, sc):
	Utils.logMessage("\n\tSuspecious account, update label and mark as training data for next round")
	pv.truncateLineCount = idx
	df.loc[idx] = refreshRecord(record, predictedLabel)
	df.to_csv(pv.mergedAccountFile, index=False, encoding='utf-8')

	removeModelFolder()
	classification.train(sc)
	cluster.train(sc)


def loadModel():
	clusterModel = KMeansModel.load(sc, pv.clusterModelPath)
	classificationModel = DecisionTreeModel.load(sc, pv.classificationModelPath)

	if pv.outputDebugMsg:
		Utils.logMessage("\nLoad cluster & classification model finished")
	return clusterModel, classificationModel


def countByFeatures(record):
	ipNum = len(str(record['buyer_ip']).split('|'))
	devIDNum = len(str(record['equipment_id']).split('|'))
	addrNum = len(record['buyer_poi'].split('|'))
	promotionNum = len(str(record['promotion_id']).split('|'))
	return (record['label'], record['buyer_pin'], ipNum ** 2, devIDNum ** 3, addrNum ** 2, promotionNum)


def calculateSim(df, matSize):
	cur = df.loc[matSize]
	ret = []
	for i in range(matSize):
		ret.append(compute.computeSimilarity(df.loc[i], cur))
	return ret


def getLabelByIdx(df,idx):
	return df.loc[idx]['label']

def refreshRecord(record, newLabel):
	retList = record.tolist()
	labelIdx = retList.index(record['label'])
	retList[labelIdx] = newLabel
	return np.array(retList)


def removeModelFolder():
	import shutil
	shutil.rmtree(pv.classificationModelPath)
	shutil.rmtree(pv.clusterModelPath)


def run():
	Utils.logTime()
	classification.preprocess()

	fp.shuffleRawData(pv.mergedAccountFile)

	Utils.logMessage("\nPretraining model started")

	cluster.train(sc)

	classification.train(sc)

	Utils.logMessage("\nPretraining model finished")
	Utils.logMessage("\nInitial accounts %s " %str(pv.truncateLineCount - 1))

	demo(10)

	Utils.logTime()

if __name__ == '__main__':
	run()
