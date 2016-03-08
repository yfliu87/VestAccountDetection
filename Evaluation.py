import codecs

def evaluateModel(model, rdd):
	'''
	Go through each record in testFile
	Predict cluster for each of them
	Output result
	'''
	sse = model.computeCost(rdd)
	print "SSE: " + str(sse)
