import codecs
import pandas as pd
import Utils


def truncate(sourceFile, targetFile, truncateLineCount):
	reader = codecs.open(sourceFile, 'r', 'gbk')
	writer = codecs.open(targetFile, 'w', 'gbk')

	count = 0
	while count < truncateLineCount:
		writer.write(reader.readline())
		count +=1

	reader.close()
	writer.close()

	Utils.logMessage("\nTruncate file finished")


def preprocess(sourceFile, targetFile, targetFields):
	reader = codecs.open(sourceFile, 'r','gbk')
	writer = codecs.open(targetFile, 'w','gbk')

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

	Utils.logMessage("\nFilter fields finished")


def filterFields(line, fieldsIdx):
	fields = line.replace('\n', '').split(',')

	ret = ''
	for i in fieldsIdx:
		ret += (fields[i] + ',')

	return ret[:-1] + '\n'


def readData(filepath):
	return pd.read_csv(filepath, encoding='gbk')


def outputNodesInSameCluster(model, unifiedRDDVecs, rawDataFrame, outputFilePath):
	centers = unifiedRDDVecs.map(lambda item: model.clusterCenters[model.predict(item)]).collect()
	rawDataFrame['center'] = centers
	groupUserByCluster(rawDataFrame).to_csv(outputFilePath, encoding='gbk')
	'''
	sorted_by_center_df = rawDataFrame.sort(columns='center')
	sorted_by_center_df.to_csv(target_file_path, encoding='gbk', index=False)
	'''
	Utils.logMessage("\nOutput cluster finished")


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
