import ClassificationMain as classification
import ClusterMain as cluster
import PredefinedValues as pv


def loadModel():
	clusterModel = KMeansModel.load(sc, pv.clusterModelPath)
	classificationModel = DecisionTreeModel.load(sc, pv.classificationModelPath)


def demo():
	sample = sc.take(10)
	predictedSample = sample.map(lambda item: (item.Label, classificationModel.predict(item.features)))
	suspeciousAccounts = predictedSample.filter(lambda item: filterSuspecious(item))
	clusteredSample = suspeciousAccounts.map(lambda item: getSimVector(item)).map(lambda item: clusterModel.centers[clusterModel.predict(item)]).collect()
	accountMap = accountsByCluster(clusteredSample)
	

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
