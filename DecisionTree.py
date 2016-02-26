import utils
from pyspark import SparkConf, SparkContext, mllib
from pyspark.mllib.tree import DecisionTree
import evaluation


def trainModel(trainingData):
	print '\nTraining Decision Tree model started'
	utils.logTime()

	model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5,maxBins=32)
	print '\nTraining Decision Tree model finished'
	utils.logTime()
	return model


def trainOptimalModel(trainingData, testData):
	print "\nTraining optimal Decision Tree model started!"
	utils.logTime()

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
					testErr, PR, ROC = evaluation.evaluate(model, testData)
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


def logMessage(optimalModel,optimalMaxDepth, optimalImpurity, optimalBinsVal, minError):

	print "\nTraining optimal Decision Tree model finished:"
	print "\tMin Test Error : " + str(minError)
	print "\toptimal impurity : " + str(optimalImpurity)
	print "\toptimal max depth : " + str(optimalMaxDepth)
	print "\toptimal bins val : " + str(optimalBinsVal)
	utils.logTime()
	#print "\toptimal model : " + optimalModel.toDebugString()


