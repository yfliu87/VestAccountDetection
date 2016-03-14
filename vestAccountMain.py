import ClassificationMain as classification
import ClusterMain as cluster
import PredefinedValues as pv


def loadModel():
	clusterModel = KMeansModel.load(sc, pv.clusterModelPath)
	classificationModel = DecisionTreeModel.load(sc, pv.classificationModelPath)


def demo():
	simMatrix = loadSimMatrix()
	sampleSize = 10
	sampleRDD = sc.take(sampleSize)
	index = [x for x in xrange(sampleSize)]
	indexedSampleRDD = sampleRDD.zip(index).map(lambda item: (item[1], item[0]))
	predictedSampleRDD = indexedSampleRDD.map(lambda item: (item[1].Label, classificationModel.predict(item[1].features)))
	suspeciousAccountsRDD = predictedSampleRDD.filter(filterSuspecious)
	clusteredSample = suspeciousAccountsRDD.map(lambda item: getSimVector(item[0], simMatrix)).map(lambda item: clusterModel.centers[clusterModel.predict(item)]).collect()
	accountMap = accountsByCluster(clusteredSample)


def loadSimMatrix():
	return pd.read_csv(pv.simMatrixFilePath)


def filterSuspecious(item):
	return item[1] >= 2 or (item[0] >= 2 and item[1] < 2)


def getSimVector(idx, simMatrix):
	return simMatrix[idx, :]


def run():
	#preprocess rule output file, mark, combine
	classification.preprocess()

	#train cluster model
	cluster.run()

	#train classification model
	classification.run()

	#load classification & cluster model to predict 
	loadModel()

	#random pick 10 records and make prediction
	demo()

if __name__ == '__main__':
	run()
