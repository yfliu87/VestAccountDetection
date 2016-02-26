from pyspark.mllib.evaluation import BinaryClassificationMetrics
import utils

def evaluate(model, testData, logMessage=False):

	predictions = model.predict(testData.map(lambda item: item.features))
	labelsAndPredictions = testData.map(lambda item: item.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda (v,p): v != p).count()/float(testData.count())
	metrics = BinaryClassificationMetrics(labelsAndPredictions)

	if logMessage:
		utils.logMessage("\nModel evaluation result:")
		utils.logMessage("\n\tTest Error: " + str(testErr))
		utils.logMessage("\n\tArea under PR = %s" %metrics.areaUnderPR)
		utils.logMessage("\n\tArea under ROC = %s" %metrics.areaUnderROC)

	return testErr, metrics.areaUnderPR, metrics.areaUnderROC
