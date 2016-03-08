import SpectralCluster as sc
import FileParser as fp
import ComputeModule as cm
import PredefinedValues as pv
import Evaluation as eva

def run():
	#truncate raw data to managable amount
	fp.truncate(pv.sourceFile, pv.truncatedFile, pv.truncateLineCount)

	fp.preprocess(pv.truncatedFile, pv.processedFile, pv.targetFields)

	rawDataFrame = fp.readData(pv.trainingFile)

	#simMat = cm.getSimilarityMatrixSerial(rawDataFrame)
	#simMat = cm.getSimilarityMatrixParallel(rawDataFrame)
	simMat = cm.getSimilarityMatrixMultiProcess(rawDataFrame)

	model, unifiedRDDVecs = sc.getClusterModel(simMat, rawDataFrame, pv.clusterNum)

	eva.evaluateModel(model, unifiedRDDVecs)

	fp.outputNodesInSameCluster(model, unifiedRDDVecs, rawDataFrame, pv.outputFile)


if __name__ == '__main__':
	run()
	
