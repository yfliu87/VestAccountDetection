#-*-coding:utf-8-*-
import codecs
import pandas as pd
import numpy as np
import Utils
import PredefinedValues as pv


def truncate(sourceFile, targetFile, truncateLineCount):
	reader = codecs.open(sourceFile, 'r','utf-8')
	writer = codecs.open(targetFile, 'w','utf-8')

	count = 0
	while count < truncateLineCount:
		writer.write(reader.readline())
		count +=1

	reader.close()
	writer.close()

	if pv.outputDebugMsg:
		Utils.logMessage("\nTruncate file finished")


def preprocess(sourceFile, targetFile, targetFields):
	reader = codecs.open(sourceFile, 'r','utf-8')
	writer = codecs.open(targetFile, 'w','utf-8')

	line = reader.readline()
	fields = line.replace('\n','').split(',')
	fieldsIdx = [] 

	for item in targetFields:
		if item in fields:
			fieldsIdx.append(fields.index(item))

	while line:
		writer.write(filterFields(line, fieldsIdx))
		line = reader.readline()

	reader.close()
	writer.close()

	if pv.outputDebugMsg:
		Utils.logMessage("\nFilter fields finished")


def filterFields(line, fieldsIdx):
	fields = line.replace('\n', '').split(',')
	ret = ''

	for i in fieldsIdx:
		ret += (fields[i] + ',')

	return ret[:-1] + '\n'


def readData(filepath):
	return pd.read_csv(filepath, encoding='utf-8')


def outputMatrix(matrix, targetFile):
	pd.DataFrame(matrix).to_csv(targetFile, header=False, index=False)


def outputNodesInSameCluster(model, unifiedRDDVecs, rawDataFrame, clusterIDCenterFilePath, clusterIDFilePath):
	centers = unifiedRDDVecs.map(lambda item: model.clusterCenters[model.predict(item)]).collect()
	rawDataFrame['clusterID'] = convertClusterID(centers)
	pd.DataFrame(rawDataFrame).to_csv(clusterIDFilePath, header=False, index=False, encoding='utf-8')

	rawDataFrame['center'] = centers
	groupUserByCluster(rawDataFrame).to_csv(clusterIDCenterFilePath, index=False, encoding='utf-8')

	if pv.outputDebugMsg:
		Utils.logMessage("\nOutput cluster finished")


def convertClusterID(centers):
	clusterIDMap = {}
	clusterId = 0
	ret = []

	for center in centers:
		scenter = center.tostring()
		if scenter not in clusterIDMap:
			clusterIDMap[scenter] = clusterId
			clusterId += 1

		ret.append(clusterIDMap[scenter])

	if pv.outputDebugMsg:
		Utils.logMessage('\nConvert to cluster ID finished')
	return ret


def groupUserByCluster(rawDataFrame):
	clusterIdxMap = {}
	rows = rawDataFrame.shape[0]

	for i in xrange(rows):
		cluster = rawDataFrame.loc[i,'center'].tostring()

		if cluster not in clusterIdxMap:
			clusterIdxMap[cluster] = []

		clusterIdxMap[cluster].append(i)

	assigned = False
	retDF = None
	for cluster, indexes in clusterIdxMap.items():
		for idx in indexes:
			if not assigned:
				retDF = pd.DataFrame(rawDataFrame.loc[idx]).T
				assigned = True
			else:
				retDF = retDF.append(rawDataFrame.loc[idx], ignore_index=True)

	return retDF


def shuffleRawData(filePath):
	df = pd.read_csv(filePath, encoding='utf-8')
	df = df.copy()
	df = df.reindex(np.random.permutation(df.index))
	df.to_csv(filePath, index=False, encoding='utf-8')
