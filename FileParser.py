import os
import codecs
import pandas as pd
import Utils

def run(sourceFile, targetFile, targetFields):
	reader = codecs.open(sourceFile, 'r','gbk')
	writer = codecs.open(targetFile, 'w','gbk')

	line = reader.readline()

	fields = line.split(',')
	fieldsIdx = [] 

	for item in targetFields:
		if item in fields:
			fieldsIdx.append(fields.index(item))

	print fieldsIdx

	while line:
		writer.write(filterFields(line, fieldsIdx))
		line = reader.readline()

	reader.close()
	writer.close()

	Utils.logMessage("\nfilter fields finished")


def filterFields(line, fieldsIdx):
	fields = line.split(',')

	ret = ''
	for i in fieldsIdx:
		ret += (fields[i] + ',')

	return ret[:-1] + '\n'


def truncate(sourceFile, targetFile,):
	reader = codecs.open(sourceFile, 'r', 'gbk')
	writer = codecs.open(targetFile, 'w', 'gbk')

	count = 0
	while count < 100:
		writer.write(reader.readline())
		count +=1

	reader.close()
	writer.close()

	Utils.logMessage("\ttruncate file finished")


def outputNodesInSameCluster(model, unifiedRDDVecs, rawdata, outputFilePath):
	Utils.logMessage("\noutput cluster started")

	df = pd.DataFrame(rawdata)
	centers = unifiedRDDVecs.map(lambda item: model.clusterCenters[model.predict(item)]).collect()
	df['center'] = centers
	df.to_csv(outputFilePath, encoding='gbk', index=False)
	'''
	sorted_by_center_df = df.sort(columns='center')
	sorted_by_center_df.to_csv(target_file_path, encoding='gbk', index=False)
	'''
	Utils.logMessage("\noutput cluster finished")


def accountAddrMap(rawDataFrame,outputFilePath):
	acct_addr_map = cm.getAccountAddressMap(rawDataFrame)
	'''
	writer = codecs.open(outputFilePath, 'w', 'gbk')

	for k,v in acct_addr_map.items():
		writer.write(k.decode('utf-8'))
		writer.write(',')
		acct_set = set(v)
		for i in acct_set:
			writer.write(i)
			writer.write(',')
			writer.write(v.count(i))
			writer.write(',')

		writer.write('\n')
	writer.close()
	'''
	return acct_addr_map


def accountDevIDMap(rawDataFrame,outputFilePath):
	acct_devid_map = cm.getAccountDevIDMap(rawDataFrame)
	'''
	writer = codecs.open(outputFilePath, 'w', 'gbk')

	for k,v in acct_devid_map.items():
		writer.write(k.decode('utf-8'))
		writer.write(',')
		acct_set = set(v)
		for i in acct_set:
			writer.write(i)
			writer.write(',')
			writer.write(v.count(i))
			writer.write(',')

		writer.write('\n')
	writer.close()
	'''
	return acct_devid_map

def readData(filepath):
	return pd.read_csv(filepath, encoding='gbk')


def getAcctDevIDAddrMap(processedFile):
	rawDataFrame = readData(processedFile)

	acct_addr_map_file = '/home/yifei/TestData/data/acct_addr_map_20160229.csv'
	acct_addr_map = accountAddrMap(rawDataFrame,acct_addr_map_file)

	acct_devid_map_file = '/home/yifei/TestData/data/acct_devid_map_20160229.csv'
	acct_devid_map = accountDevIDMap(rawDataFrame,acct_devid_map_file)

	writer = codecs.open('/home/yifei/TestData/data/acct_dev_addr_20160229.csv','w','gbk')
	for addr, acct in acct_addr_map.items():
		for devid, acct1 in acct_devid_map.items():
			intersection = set(acct).intersection(set(acct1))
			if len(intersection) > 1:
				writer.write(addr.decode('utf-8'))
				writer.write(',')
				writer.write(devid)
				writer.write(',')
				for i in intersection:
					writer.write(i)
					writer.write(',')
				
				writer.write('\n')

	writer.close()
