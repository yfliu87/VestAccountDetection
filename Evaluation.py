#-*-coding:utf-8-*-
from pyspark.mllib.evaluation import MulticlassMetrics
import Utils
import PredefinedValues as pv

def evaluateModel(model, rdd):
	sse = model.computeCost(rdd)
	return sse

def selfDefinedError(label, prediction):
	if label == prediction:
		return (False)
	elif (label == 3 and prediction == 2) or (label == 2 and prediction == 3) or (label == 1 and prediction ==0) or (label == 0 and prediction == 1):
		return (False)
	else:
		return (True)


def calculateErrorRate(msg, rdd):
	error = rdd.filter(lambda(v,p): selfDefinedError(v, p)).count()/float(rdd.count())

	Utils.logMessage(msg + " measurements")
	Utils.logMessage("\tPrecision: " + str(1 - error))
	

def clusterModelMeasurements(msg, predictionLabelPair):
	metrics = MulticlassMetrics(predictionLabelPair)

	Utils.logMessage("\nCluster model " + msg + " measurements")
	Utils.logMessage("\tPrecision: " + str(metrics.precision()))


def classificationModelMeasurements(msg, predictionLabelPair):
	metrics = MulticlassMetrics(predictionLabelPair)

	Utils.logMessage("\nClassification model " + msg + " measurements")
	Utils.logMessage("\n\tPrecision: " + str(metrics.precision()))
	Utils.logMessage("\n\tRecall: " + str(metrics.recall()))
	Utils.logMessage("\n\tf measure: " + str(metrics.fMeasure()))

	if pv.isLastRecord:
		Utils.logMessage("\n\tFalse positive rate of 0: " + str(metrics.falsePositiveRate(0.0)))
		Utils.logMessage("\n\tFalse positive rate of 1: " + str(metrics.falsePositiveRate(1.0)))
		Utils.logMessage("\n\tFalse positive rate of 2: " + str(metrics.falsePositiveRate(2.0)))
		Utils.logMessage("\n\tFalse positive rate of 3: " + str(metrics.falsePositiveRate(3.0)))
		Utils.logMessage("\n\tTrue positive rate of 0: " + str(metrics.truePositiveRate(0.0)))
		Utils.logMessage("\n\tTrue positive rate of 1: " + str(metrics.truePositiveRate(1.0)))
		Utils.logMessage("\n\tTrue positive rate of 2: " + str(metrics.truePositiveRate(2.0)))
		Utils.logMessage("\n\tTrue positive rate of 3: " + str(metrics.truePositiveRate(3.0)))
