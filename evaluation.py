from pyspark.mllib.evaluation import BinaryClassificationMetrics

def evaluate(model, testData):
	predictions = model.predict(testData.map(lambda item: item.features))
	labelsAndPredictions = testData.map(lambda item: item.label).zip(predictions)

	predictionAndLabels = testData.map(lambda item: (float(model.predict(item.features)), item.label))
	print "Test Error: " + str(predictionAndLabels.filter(lambda (p,v): p != v).count()/float(testData.count()))

	metrics = BinaryClassificationMerics(predictionAndLabels)
	print "\nArea under PR = %s" %metrics.areaUnderPR
	print "\nArea under ROC = %s" %metrics.areaUnderROC

