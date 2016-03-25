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
counter = 0
isNewModel = False

def demo(count):
	global counter, isNewModel
	df = None

	for idx in xrange(pv.truncateLineCount + 1, pv.truncateLineCount + count + 1):
		Utils.logMessage("\nUser %s" %str(idx))

		if isNewModel:
			clusterModel, classificationModel = loadModel()
			df = fp.readData(pv.mergedAccountFile)
			isNewModel = False

		counter += 1
		record = df.loc[idx]

		compressedRecord = countByFeatures(record)

		labeledRecord = LabeledPoint(compressedRecord[0], compressedRecord[2:])

		predictedLabel = classificationModel.predict(labeledRecord.features)

		if predictedLabel <= 1:
			Utils.logMessage("\nPredicted label: %s, safe account, go for next" %str(predictedLabel))
			df = updateLabelForNextRoundTrain(idx, df, record, predictedLabel, sc, counter, "\n\tRecalculate sim matrix")
		else:
			Utils.logMessage("\nPredicted label: %s, risky account, double check using cluster model" %str(predictedLabel))

			sim = compute.getSimilarityByRDD(sc, df[:idx])

			if max(sim) > pv.simThreshold:
				mostSimilarAccountIdx = sim.index(max(sim))
				labelFromMostSimilarAccount = getLabelByIdx(df, mostSimilarAccountIdx)

				Utils.logMessage("\n\tLabel of most similar account is %s" %str(labelFromMostSimilarAccount))

				if labelFromMostSimilarAccount > predictedLabel:
					df = updateLabelForNextRoundTrain(idx, df, record, labelFromMostSimilarAccount, sc, counter, "\n\tSuspecious account, update label and mark as training data for next round %s" %str(idx))
				else:
					df = updateLabelForNextRoundTrain(idx, df, record, predictedLabel, sc, counter, "\n\tRecalculate sim matrix")
			else:
				df = updateLabelForNextRoundTrain(idx, df, record, predictedLabel, sc, counter,"\nNo similar account found, recalculate sim matrix")


	Utils.logMessage("Job Finished!")


def updateLabelForNextRoundTrain(idx, df, record, newLabel, sc, counter, msg):
	Utils.logMessage(msg)
	pv.truncateLineCount = idx
	df.loc[idx] = refreshRecord(record, newLabel)

	if counter % 1000 == 0:
		global isNewModel
		isNewModel = True
		removeModelFolder()
		df.to_csv(pv.mergedAccountFile, index=False, encoding='utf-8')
		classification.trainWithDF(sc, df[0:idx-1])
		cluster.trainWithDF(sc, df[0:idx-1])

	return df


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

	global isNewModel
	isNewModel = True
	Utils.logMessage("\nPretraining model finished")
	Utils.logMessage("\nInitial accounts %s " %str(pv.truncateLineCount - 1))

	pv.isTrainingRound = False
	demo(10000)

	Utils.logTime()

if __name__ == '__main__':
	run()
