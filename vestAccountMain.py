import ClassificationMain as classification
import ClusterMain as cluster
import PredefinedValues as pv


def loadModel():
	clusterModel = KMeansModel.load(sc, pv.clusterModelPath)
	classificationModel = DecisionTreeModel.load(sc, pv.classificationModelPath)


def demo():
	simMatrix = loadSimMatrix()
	sample = sc.take(10)
	predictedSample = sample.map(lambda item: (item.Label, classificationModel.predict(item.features)))
	suspeciousAccounts = predictedSample.filter(filterSuspecious)
	clusteredSample = suspeciousAccounts.map(lambda item: getSimVector(item, simMatrix)).map(lambda item: clusterModel.centers[clusterModel.predict(item)]).collect()
	accountMap = accountsByCluster(clusteredSample)


def loadSimMatrix():
	return pd.read_csv(pv.simMatrixFilePath)


def filterSuspecious(item):
	return item[1] >= 2 or (item[0] >= 2 and item[1] < 2)


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
