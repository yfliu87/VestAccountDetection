#-*-coding:utf-8-*-

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

def calculateErrorRate(rdd):
	error = rdd.filter(lambda (v,p): selfDefinedError(v,p)).count()/float(rdd.count())
	return error

