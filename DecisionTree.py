
from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.tree import DecisionTree


def trainModel(trainingData):
	model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5,maxBins=32)
	print '\nTraining DecisionTree model finished'
	return model


def trainOptimalModel(trainingData, testData):
	impurityVals = ['gini', 'entropy']
	maxDepthVals = [3,4,5,6,7]
	maxBinsVals = [8,16,32]

	optimalModel = None
	optimalMaxDepth = None
	optimalImpurity = None
	optimalBinsVal = None
	minError = None

	try:
		for curImpurity in impurityVals:
			for curMaxDepth in maxDepthVals:
				for curMaxBins in maxBinsVals:
					model = DecisionTree.trainClassifier(trainingData, 
														 numClasses=2, 
														 categoricalFeaturesInfo={}, 
														 impurity=curImpurity, 
														 maxDepth=curMaxDepth,
														 maxBins=curMaxBins)
					testErr = evaluateModel(model, testData)
					if testErr < minError or not minError:
						minError = testErr
						optimalImpurity = curImpurity
						optimalMaxDepth = curMaxDepth
						optimalBinsVal = curMaxBins
						optimalModel = model
	except:
		print "\nException during model training with below parameters:"
		print "\timpurity: " + str(curImpurity)
		print "\tmaxDepth: " + str(curMaxDepth)
		print "\tmaxBins: " + str(curMaxBins)

	logMessage(optimalModel, optimalMaxDepth, optimalImpurity, optimalBinsVal, minError)
	return optimalModel 


def evaluateModel(model, testData):
	predictions = model.predict(testData.map(lambda item: item.features))
	labelsAndPredictions = testData.map(lambda item: item.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda (v,p): v != p).count()/float(testData.count())
	return testErr

def logMessage(optimalModel,optimalMaxDepth, optimalImpurity, optimalBinsVal, minError):

	print "\nOptimal DecisionTree Model :"
	print "\tMin Test Error : " + str(minError)
	print "\toptimal impurity : " + str(optimalImpurity)
	print "\toptimal max depth : " + str(optimalMaxDepth)
	print "\toptimal bins val : " + str(optimalBinsVal)
	#print "\toptimal model : " + optimalModel.toDebugString()


