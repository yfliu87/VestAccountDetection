#-*-coding:utf-8-*-

def evaluateModel(model, rdd):
	print "SSE: " + str(model.computeCost(rdd))

def calculateErrorRate(rdd):
	error = rdd.filter(lambda (v,p): v != p).count()/float(rdd.count())
	print "\nError rate: " + str(error)
