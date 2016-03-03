import SpectralCluster as sc
import FileParser as fp
import ComputeModule as cm
import PredefinedValues as preVal
import Evaluation as eva


def run(trainingFilePath, testFilePath, outputFilePath):
	rawDataFrame = fp.readData(trainingFilePath)

	simMat = cm.getSimilarityMatrix(rawDataFrame)

	model, unifiedRDDVecs = sc.getClusters(simMat, rawDataFrame, outputFilePath, 20)

	eva.evaluateModel(model, testFilePath)

	fp.outputNodesInSameCluster(model, unifiedRDDVecs, rawDataFrame, outputFilePath)


if __name__ == '__main__':

	sourceFile ='/home/yifei/TestData/data/order_raw_20160229.csv' 
	truncatedFile = '/home/yifei/TestData/data/order_truncated_20160229.csv'
	fp.truncate(sourceFile,truncatedFile)	

	processedFile = '/home/yifei/TestData/data/order_processed_20160229.csv'
	fp.preprocess(truncatedFile, processedFile)

	#train cluster model
	trainingFile = processedFile
	testFile = ''
	outputFile = '/home/yifei/TestData/data/order_clustered_20160229.csv'
	run(trainingFile, '', outputFile)

	#output account devID address map
	outputMapFile = '/home/yifei/TestData/data/acct_dev_addr_20160229.csv'
	fp.writeAcctDevIDAddrMap(processedFile, outputMapFile)
