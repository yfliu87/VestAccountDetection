#-*-coding:utf-8-*-
import pandas as pd
import ClassificationMain as classification
import ClusterMain as cluster
import PredefinedValues as pv
import FileParser as fp
import Utils


def loadModel():
	clusterModel = KMeansModel.load(sc, pv.clusterModelPath)
	classificationModel = DecisionTreeModel.load(sc, pv.classificationModelPath)
	Utils.logMessage("\nLoad cluster & classification model finished")


def demo(count):

	for idx in xrange(count):
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
			sim = calculateSim(record, loadSimMatrix())

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


def loadSimMatrix():
	return pd.read_csv(pv.simMatrixFilePath)


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
