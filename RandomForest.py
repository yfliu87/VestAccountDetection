import utils
from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.tree import RandomForest

def trainModel(trainingData):
	print "\nTrainning Random Forest model started!"
	utils.logTime()

	model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, 
											numTrees=3, featureSubsetStrategy="auto", impurity='gini',
											maxDepth=5, maxBins=32)

	print '\nTraining Random Forest model finished'
	utils.logTime()
	return model


def trainOptimalModel(trainingData, testData):
	print "\nTraining optimal Random Forest model started!"
	utils.logTime()

	numTreesVals = [3,5,8]
	featureSubsetStrategyVals = ['auto','all','sqrt','log2','onethird']
	impurityVals = ['gini', 'entropy']
	maxDepthVals = [3,4,5,6,7]
	maxBinsVals = [8,16,32]

	optimalModel = None
	optimalNumTrees = None
	optimalFeatureSubsetStrategy = None
	optimalMaxDepth = None
	optimalImpurity = None
	optimalBinsVal = None
	minError = None

	try:
		for curNumTree in numTreesVals:
			for curFeatureSubsetStrategy in featureSubsetStrategyVals:
				for curImpurity in impurityVals:
					for curMaxDepth in maxDepthVals:
						for curMaxBins in maxBinsVals:
							model = RandomForest.trainClassifier(trainingData, 
																numClasses=2, 
																categoricalFeaturesInfo={}, 
														 		numTrees=curNumTree,
														 		featureSubsetStrategy=curFeatureSubsetStrategy,
														 		impurity=curImpurity, 
														 		maxDepth=curMaxDepth,
														 		maxBins=curMaxBins)
							testErr = evaluateModel(model, testData)
							if testErr < minError or not minError:
								minError = testErr
								optimalNumTrees = curNumTree
								optimalFeatureSubsetStrategy = curFeatureSubsetStrategy
								optimalImpurity = curImpurity
								optimalMaxDepth = curMaxDepth
								optimalBinsVal = curMaxBins
								optimalModel = model
	except:
		print "\nException during model training with below parameters:"
		print "\tnum trees: " + str(optimalNumTrees)
		print "\tfeature subset strategy: " + optimalFeatureSubsetStrategy
		print "\timpurity: " + str(curImpurity)
		print "\tmaxDepth: " + str(curMaxDepth)
		print "\tmaxBins: " + str(curMaxBins)

	logMessage(optimalModel, optimalNumTrees, optimalFeatureSubsetStrategy, optimalMaxDepth, optimalImpurity, optimalBinsVal, minError)
	return optimalModel 


def evaluateModel(model, testData):
	predictions = model.predict(testData.map(lambda item: item.features))
	labelsAndPredictions = testData.map(lambda item: item.label).zip(predictions)
	return labelsAndPredictions.filter(lambda (v,p): v != p).count()/float(testData.count())


def logMessage(optimalModel,optimalNumTrees, optimalFeatureSubsetStrategy, optimalMaxDepth, optimalImpurity, optimalBinsVal, minError):

	print "\nTraining optimal Random Forest model finished:"
	print "\tMin Test Error : " + str(minError)
	print "\toptimal num trees : " + str(optimalNumTrees)
	print "\toptimal feature subset strategy: " + str(optimalFeatureSubsetStrategy)
	print "\toptimal impurity : " + str(optimalImpurity)
	print "\toptimal max depth : " + str(optimalMaxDepth)
	print "\toptimal bins val : " + str(optimalBinsVal)
	utils.logTime()
	#print "\toptimal model : " + optimalModel.toDebugString()