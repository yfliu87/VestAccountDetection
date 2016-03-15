#-*-coding:utf-8-*-
import pandas as pd
import ClassificationMain as classification
import ClusterMain as cluster
import PredefinedValues as pv
import FileParser as fp
import Utils

clusterAccountMap = {}

def demo(count):
	for idx in xrange(pv.truncateLineCount, pv.truncateLineCount + count):
		loadModel()

		#take random account from merged file
		df = fp.readData(pv.mergedAccountFile)
		record = df[idx,:]

		#organize feature count
		compressedRecord = classification.countByFeatures(record)
		labeledRecord = LabelPoint(compressedRecord[0], compressedRecord[1:])

		#predict cluster
		predictedLabel = classificationModel.predict(labeledRecord.features)

		if predictedLabel <= 1:
			print "\nPredicted label: %s, safe account, go for next" %str(predictedLabel)
			continue

		else:
			print "\nPredicted label: %s, risky account, double check using cluster model" %str(predictedLabel)

			#calculate similarity with existing simMatrix
			sim = calculateSim(pv.truncateLineCount + idx)

			#find accounts within same cluster 
			predictedCluster = clusterModel.centers[clusterModel.predict(sim)]

			similarAccounts = getClusterAccounts(predictedCluster, clusterAccountMap)

			newLabel = checkSimilarAccounts(similarAccounts)

			if newLabel >= predictedLabel:
				print "\nSuspecious account, mark as training data for next round"

				#retrain classification model


				#retrain cluster model

			else:
				print "\nLow risk account, go for next"


def loadModel():
	clusterModel = KMeansModel.load(sc, pv.clusterModelPath)
	buildClusterAccountMap(pv.clusterIDCenterFile)
	classificationModel = DecisionTreeModel.load(sc, pv.classificationModelPath)
	Utils.logMessage("\nLoad cluster & classification model finished")

def buildClusterAccountMap(clusterAccountFilePath):
	df = fp.readData(clusterAccountFilePath)
	for idx in xrange(df.shape[0]):
		pin = df['buyer_pin']
		cluster = df['center']

		if cluster not in clusterAccountMap:
			clusterAccountMap[cluster] = []

		clusterAccountMap[cluster].append(pin)


def calculateSim(df, matSize):
	return cm.getSimilarityMatrixMultiProcess(df[:matSize])


def getClusterAccounts(cluster, clusterAccountMap):
	return clusterAccountMap[cluster]

def run():
	#preprocess rule output file, mark, combine
	classification.preprocess()

	#train cluster model
	cluster.run()

	#train classification model
	classification.run()

	#random pick 10 records and make prediction
	demo(10)

if __name__ == '__main__':
	run()
