import numpy as np
import Utils
import PredefinedValues as preVal
import PredefinedValues.DEFAULT_SIMILARITY as DEFAULTSIM


def getAccountAddressMap(rawDataFrame):
	acct_addr_map = {}
	rows = rawDataFrame.shape[0]
	for i in xrange(rows):
		record = rawDataFrame.loc[i]

		account = record['buyer_pin']
		address = combineAdd(record['buyer_city_name'],record['buyer_country_name'],record['buyer_poi'])

		if address not in acct_addr_map:
			acct_addr_map[address] = []

		acct_addr_map[address].append(account)
	return acct_addr_map

def getAccountDevIDMap(rawDataFrame):
	acct_devid_map = {}
	rows = rawDataFrame.shape[0]
	for i in xrange(rows):
		record = rawDataFrame.loc[i]

		account = record['buyer_pin']
		devid = record['equipment_id']

		if devid not in acct_devid_map:
			acct_devid_map[devid] = []

		acct_devid_map[devid].append(account)
	return acct_devid_map


def getSimilarityMatrix(rawDataFrame):
	Utils.logMessage("\nbuild similarity matrix started")

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
				sim = computeSimilarity(rawDataFrame.loc[i], rawDataFrame.loc[j])
				resultHash[(i,j)] = sim

			print sim
			simVector.append(sim)

		simMat.append(simVector)

	Utils.logMessage("build similarity matrix finished")
	return np.matrix(np.array(simMat))


def computeSimilarity(user1, user2):
	simWeight = preVal.simWeight

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

	'''
	add1 = combineAdd(user1['buyer_city_name'],user1['buyer_country_name'],user1['buyer_poi'])
	add2 = combineAdd(user2['buyer_city_name'],user2['buyer_country_name'],user2['buyer_poi'])

	city_county_poi_sim = computeAddrSim(add1, add2)

	return (simWeight['buyer_ip']*ip_sim + simWeight['buyer_mobile']*tel_sim 
	+ simWeight['buyer_full_address']*city_county_poi_sim)
	'''
	

def combineAdd(city, county, poi):
	ret = ''
	if not isinstance(city,float):
		ret += city.encode('utf-8')
	
	if not isinstance(county,float):
		ret += county.encode('utf-8')

	if not isinstance(poi,float):
		ret += poi.encode('utf-8')

	return ret


def computeIPSim(ips1, ips2):
	try:
		intersection = set(ips1).intersection(set(ips2))
		union = set(ips1).union(set(ips2))
		return len(intersection)/float(len(union))
	except:
		return DEFAULTSIM


def computeTelSim(tels1, tels2):
	try:
		intersection = set(tels1).intersection(set(tels2))
		union = set(tels1).union(set(tels2))
		return len(intersection)/float(len(union))
	except:
		return DEFAULTSIM


def computeAddrSim(adds1, adds2):
	try:
		intersection = set(adds1).intersection(set(adds2))
		union = set(adds1).union(set(adds2))
		return len(intersection)/float(len(union))
	except:
		return DEFAULTSIM


def computeDevIDSim(devIDs1, devIDs2):
	try:
		intersection = set(devIDs1).intersection(set(devIDs2))
		union = set(devIDs1).union(set(devIDs2))
		return len(intersection)/float(len(union))
	except:
		return DEFAULTSIM


def computeRecSim(recs1, recs2):
	try:
		intersection = set(recs1).intersection(set(recs2))
		union = set(recs1).union(set(recs2))
		return len(intersection)/float(len(union))
	except:
		return DEFAULTSIM


def computeCitySim(cities1, cities2):
	try:
		if not isinstance(cities1,float) and not isinstance(cities2, float):
			intersection = set(list(cities1.encode('utf-8'))).intersection(set(list(cities2.encode('utf-8'))))
			union = set(list(cities1.encode('utf-8'))).union(set(list(cities2.encode('utf-8'))))
			return len(intersection)/float(len(union))
		else:
			return DEFAULTSIM
	except:
		return DEFAULTSIM


def computeCountySim(county1, county2):
	try:
		if not isinstance(county1, float) and not isinstance(county2):
			intersection = set(list(county1.encode('utf-8'))).intersection(set(list(county2.encode('utf-8'))))
			union = set(list(county1.encode('utf-8'))).union(set(list(county2.encode('utf-8'))))
			return len(intersection)/float(len(union))
		else:
			return DEFAULTSIM
	except:
		return DEFAULTSIM


def computePoiSim(poi1, poi2):
	try:
		if not isinstance(county1, float) and not isinstance(county2):
			intersection = set(list(poi1.encode('utf-8'))).intersection(set(list(poi2.encode('utf-8'))))
			union = set(list(poi1.encode('utf-8'))).union(set(list(poi2.encode('utf-8'))))
			return len(intersection)/float(len(union))
		else:
			return DEFAULTSIM
	except:
		return DEFAULTSIM