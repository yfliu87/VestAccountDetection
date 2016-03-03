import numpy as np
import pandas as pd
import codecs
import SpectralCluster as sc
import FileParser as fp
import ComputeModule as cm
import PredefinedValues as preVal
import Evaluation as eva


def run(trainingFilePath, outputFilePath):
	rawDataFrame = fp.readData(trainingFilePath)

	simMat = cm.getSimilarityMatrix(rawDataFrame)

	model = sc.getClusters(simMat, rawDataFrame, outputFilePath, 200)

	eva.evaluateModel(model, testFilePath)

	fp.outputNodesInSameCluster(model, unifiedRDDVecs, rawdata, outputFilePath)


if __name__ == '__main__':

	sourceFile ='/home/yifei/TestData/data/order_raw_20160229.csv' 
	truncatedFile = '/home/yifei/TestData/data/order_truncated_20160229.csv'
	fp.shorten(sourceFile,truncatedFile)	

	processedFile = '/home/yifei/TestData/data/order_processed_20160229.csv'
	fp.run(truncatedFile, processedFile, targetFields)

	#train cluster model
	run(processedFile, '/home/yifei/TestData/data/order_clustered_20160229.csv')

	#output account devID address map
	fp.getAcctDevIDAddrMap()
