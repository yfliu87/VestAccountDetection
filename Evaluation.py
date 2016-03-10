#-*-coding:utf-8-*-

def evaluateModel(model, rdd):
	sse = model.computeCost(rdd)
	print "\nSSE: " + str(sse)
	return sse

def calculateErrorRate(rdd):
	error = rdd.filter(lambda (v,p): v != p).count()/float(rdd.count())
	print "\nError rate: " + str(error)
	return error
