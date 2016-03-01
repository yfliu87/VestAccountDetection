import utils
import numpy as np

def getSimilarityMatrix(rawDataFrame, simWeight):
	utils.logMessage("\nbuild similarity matrix started")

	resultHash = {}
	simMat = []

	rows = rawDataFrame.shape[0]
	for i in xrange(rows):
		simVector = []
		for j in xrange(rows):
			sim = 0.0
			if i == j:
				sim = 100
			elif i > j:
				sim = resultHash[(j,i)]
			else:
				sim = computeSimilarity(rawDataFrame.loc[i], rawDataFrame.loc[j], simWeight)
				resultHash[(i,j)] = sim

			simVector.append(sim)

		simMat.append(simVector)

	utils.logMessage("build similarity matrix finished")
	return np.matrix(np.array(simMat))


def computeSimilarity(user1, user2, simWeight):
	ip_sim = computeIPSim(user1['buyer_ip'], user2['buyer_ip'])
	tel_sim = computeTelSim(user1['buyer_mobile'], user2['buyer_mobile'])
	address_sim = computeAddrSim(user1['buyer_full_address'], user2['buyer_full_address'])
	deviceID_sim = computeDevIDSim(user1['equipment_id'], user2['equipment_id'])
	receiver_sim = computeRecSim(user1['buyer_full_name'], user2['buyer_full_name'])
	city_sim = computeCitySim(user1['buyer_city_name'], user2['buyer_city_name'])
	county_sim = computeCountySim(user1['buyer_country_name'], user2['buyer_country_name'])
	poi_sim = computePoiSim(user1['buyer_poi'], user2['buyer_poi'])

	return (simWeight['buyer_ip']*ip_sim + simWeight['buyer_mobile']*tel_sim 
	+ simWeight['buyer_full_address']*address_sim + simWeight['equipment_id']*deviceID_sim 
	+ simWeight['buyer_full_name']*receiver_sim + simWeight['buyer_city_name']*city_sim 
	+ simWeight['buyer_county_name']*county_sim + simWeight['buyer_poi']*poi_sim)


def computeIPSim(ips1, ips2):
	try:
		intersection = list(set(ips1).intersection(set(ips2)))
		union = list(set(ips1).union(set(ips2)))
		return len(intersection)/float(len(union))
	except:
		return 0.0


def computeTelSim(tels1, tels2):
	try:
		intersection = list(set(tels1).intersection(set(tels2)))
		union = list(set(tels1).union(set(tels2)))
		return len(intersection)/float(len(union))
	except:
		return 0.0


def computeAddrSim(adds1, adds2):
	try:
		intersection = list(set(adds1).intersection(set(adds2)))
		union = list(set(adds1).union(set(adds2)))
		return len(intersection)/float(len(union))
	except:
		return 0.0


def computeDevIDSim(devIDs1, devIDs2):
	try:
		intersection = list(set(devIDs1).intersection(set(devIDs2)))
		union = list(set(devIDs1).union(set(devIDs2)))
		return len(intersection)/float(len(union))
	except:
		return 0.0


def computeRecSim(recs1, recs2):
	try:
		intersection = list(set(recs1).intersection(set(recs2)))
		union = list(set(recs1).union(set(recs2)))
		return len(intersection)/float(len(union))
	except:
		return 0.0


def computeCitySim(cities1, cities2):
	try:
		intersection = list(set(cities1).intersection(set(cities2)))
		union = list(set(cities1).union(set(cities2)))

		return len(intersection)/float(len(union))
	except:
		return 0.0


def computeCountySim(counties1, counties2):
	try:
		intersection = list(set(counties1).intersection(set(counties2)))
		union = list(set(counties1).union(set(counties2)))
		return len(intersection)/float(len(union))
	except:
		return 0.0


def computePoiSim(pois1, pois2):
	try:
		intersection = list(set(pois1).intersection(set(pois2)))
		union = list(set(pois1).union(set(pois2)))
		return len(intersection)/float(len(union))
	except:
		return 0.0