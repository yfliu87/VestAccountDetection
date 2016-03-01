import numpy as np
import pandas as pd
import SpectralCluster
import RawOrderParser as rop
import ComputeModule as cm
import utils

targetFields = ['order_id','buyer_full_name','buyer_full_address','buyer_mobile','buyer_ip','equipment_id','buyer_city_name','buyer_country_name','buyer_poi']

simWeight = {'buyer_ip':10, 
			 'buyer_mobile':10, 
			 'buyer_full_address':10, 
			 'equipment_id': 10, 
			 'buyer_full_name':10,
			 'buyer_city_name':10,
			 'buyer_county_name':20,
			 'buyer_poi':20}


def run(inputFilePath, outputFilePath):
	rawDataFrame = readData(inputFilePath)

	simMat = cm.getSimilarityMatrix(rawDataFrame, simWeight)

	SpectralCluster.getClusters(simMat, rawDataFrame, outputFilePath, 200)


def readData(filepath):
	return pd.read_csv(filepath, encoding='gbk')



if __name__ == '__main__':

	sourceFile ='/home/yifei/TestData/data/order_raw_20160229.csv' 
	filteredFile = '/home/yifei/TestData/data/order_filtered_20160229.csv'
	rop.shorten(sourceFile,filteredFile)	

	processedFile = '/home/yifei/TestData/data/order_processed_20160229.csv'
	rop.run(filteredFile, processedFile, targetFields)

	outputFilePath = '/home/yifei/TestData/data/order_clustered_20160229.csv'
	run(processedFile, outputFilePath)