import numpy as np
import pandas as pd
import codecs
import SpectralCluster
import RawOrderParser as rop
import ComputeModule as cm
import PredefinedValues as preVal


def run(inputFilePath, outputFilePath):
	rawDataFrame = readData(inputFilePath)

	simMat = cm.getSimilarityMatrix(rawDataFrame)

	SpectralCluster.getClusters(simMat, rawDataFrame, outputFilePath, 200)


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


if __name__ == '__main__':

	sourceFile ='/home/yifei/TestData/data/order_raw_20160229.csv' 
	#truncatedFile = '/home/yifei/TestData/data/order_truncated_20160229.csv'
	#rop.shorten(sourceFile,truncatedFile)	

	processedFile = '/home/yifei/TestData/data/order_processed_20160229.csv'
	rop.run(sourceFile, processedFile, targetFields)

	#train cluster model
	run(processedFile, '/home/yifei/TestData/data/order_clustered_20160229.csv')

	#output account devID address map
	#getAcctDevIDAddrMap()
