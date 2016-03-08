import SpectralCluster as sc
import FileParser as fp
import ComputeModule as cm
import PredefinedValues as preVal
import Evaluation as eva

'''
sourceFile ='/home/yifei/TestData/data/order_raw_20160229.csv' 
truncatedFile = '/home/yifei/TestData/data/order_truncated_20160229.csv'
processedFile = '/home/yifei/TestData/data/order_processed_20160229.csv'
trainingFile = processedFile
testFile = ''
outputFile = '/home/yifei/TestData/data/order_clustered_20160229.csv'
outputMapFile = '/home/yifei/TestData/data/acct_dev_addr_20160229.csv'
clusters = 400
'''

sourceFile ='/home/yifei/TestData/data/realdata/testdata_20160307.csv' 
truncatedFile = '/home/yifei/TestData/data/realdata/testdata_truncated_20160307.csv'
processedFile = '/home/yifei/TestData/data/realdata/testdata_processed_20160307.csv'
trainingFile = processedFile
testFile = ''
outputFile = '/home/yifei/TestData/data/realdata/testdata_clustered_20160307.csv'
clusters = 30

def run():
	#truncate raw data to managable amount
	fp.truncate(sourceFile,truncatedFile)	

	#filter out interested fields from truncated file
	fp.preprocess(truncatedFile, processedFile)

	rawDataFrame = fp.readData(trainingFile)

	#simMat = cm.getSimilarityMatrixSerial(rawDataFrame)
	#simMat = cm.getSimilarityMatrixParallel(rawDataFrame)
	simMat = cm.getSimilarityMatrixMultiProcess(rawDataFrame)

	model, unifiedRDDVecs = sc.getClusterModel(simMat, rawDataFrame, clusters)

	eva.evaluateModel(model, unifiedRDDVecs)

	fp.outputNodesInSameCluster(model, unifiedRDDVecs, rawDataFrame, outputFile)

	#output account devID address map
	#fp.writeAcctDevIDAddrMap(processedFile, outputMapFile)


if __name__ == '__main__':
	run()
	
