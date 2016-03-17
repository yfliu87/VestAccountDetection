#-*-coding:utf-8-*-
import numpy as np
import Utils
import PredefinedValues as pv 


def getSimilarityMatrixMultiProcess(rawDataFrame):
	from multiprocessing import Pool
	rows = rawDataFrame.shape[0]

	if pv.outputDebugMsg:
		Utils.logMessage("\nBuild similarity matrix of size %d x %d started" %(rows, rows))
		Utils.logTime()

	indexes = [i for i in xrange(rows)]
	simMat = []
	pool = Pool(4)
	for idx in indexes:
		simMat.append(pool.apply(computeSim, (idx, rawDataFrame)))

	pool.close()
	pool.join()

	if pv.outputDebugMsg:
		Utils.logMessage("\nBuild similarity matrix finished")
		Utils.logTime()

	mat = np.matrix(simMat)

	return np.add(mat, mat.T)
	

def computeSim(rowIdx, rawDataFrame):
	rows = rawDataFrame.shape[0]
	simVector = []

	for i in xrange(rows):
		sim = 0.0
		if i == rowIdx:
			sim = 50
		elif i < rowIdx:
			sim = 0.0
		else:
			sim = computeSimilarity(rawDataFrame.loc[i], rawDataFrame.loc[rowIdx])

		simVector.append(sim)

	return simVector


def computeSimilarity(user1, user2):
	simWeight = pv.simWeight

	ip_sim = computeIPSim(user1['buyer_ip'], user2['buyer_ip'])
	deviceID_sim = computeDevIDSim(user1['equipment_id'], user2['equipment_id'])
	poi_sim = computePoiSim(user1['buyer_poi'], user2['buyer_poi'])
	promotion_sim = computePromotionSim(user1['promotion_id'], user2['promotion_id'])
	return (simWeight['buyer_ip']*ip_sim + simWeight['equipment_id']*deviceID_sim 
		+ simWeight['buyer_poi']*poi_sim + simWeight['promotion_id']*promotion_sim)

	
def computePromotionSim(promotions1, promotions2):
	try:
		proms1 = promotions1.split('|')
		proms2 = promotions2.split('|')
		intersection = set(proms1).intersection(set(proms2))
		union = set(proms1).union(set(proms2))
		return len(intersection)/float(len(union))
	except:
		print "Promotion exception"
		return pv.DEFAULTSIM


def computeIPSim(buyer_ips1, buyer_ips2):
	try:
		ips1 = buyer_ips1.split('|')
		ips2 = buyer_ips2.split('|')
		intersection = set(ips1).intersection(set(ips2))
		union = set(ips1).union(set(ips2))
		return len(intersection)/float(len(union))
	except:
		print "IP exception"
		return pv.DEFAULTSIM


def computeDevIDSim(devIDs1, devIDs2):
	try:
		if isinstance(devIDs1, unicode) and isinstance(devIDs2, unicode):
			devids1 = devIDs1.split('|')
			devids2 = devIDs2.split('|')
			intersection = set(devids1).intersection(set(devids2))
			union = set(devids1).union(set(devids2))
			return len(intersection)/float(len(union))
		else:
			return pv.DEFAULTSIM
	except:
		print "DevID exception"
		return pv.DEFAULTSIM


def computePoiSim(poi1, poi2):
	try:
		pois1 = poi1.split('|')
		pois2 = poi2.split('|')

		fullAdd1 = []
		fullAdd2 = []
		
		for item in pois1:
			fullAdd1 += item.split('_')

		for item in pois2:
			fullAdd2 += item.split('_')

		intersection = set(fullAdd1).intersection(set(fullAdd2))
		union = set(fullAdd1).union(set(fullAdd2))
		return len(intersection)/float(len(union))

	except:
		print "POI exception"
		return pv.DEFAULTSIM
