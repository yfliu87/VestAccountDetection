#-*-coding:utf-8-*-
from __future__ import division
from scipy import sparse
from scipy.sparse.linalg.eigen import arpack
from numpy import *
from pyspark.mllib.clustering import KMeans, KMeansModel
import Utils
import FileParser as fp


def getClusterModel(sc, mat, rawdata, clusterNum, dimensionReductionNum, targetEigenVecFile):
	laplacianMat = getLaplacianMatrix(mat)

	vals, vecs = computeEigenValsVectors(laplacianMat, dimensionReductionNum)

	unifiedEigenVec = unification(vecs)

	fp.outputMatrix(unifiedEigenVec, targetEigenVecFile)

	unifiedRDDVecs = sc.parallelize(unifiedEigenVec)

	model = kMeans(unifiedRDDVecs,clusterNum)

	Utils.logMessage("\nCluster finished") 

	return model, unifiedRDDVecs


def getLaplacianMatrix(mat):
	D = mat.sum(1)
	D = sqrt(1/D)
	n = len(D)
	D = D.T
	D = sparse.spdiags(D, 0, n, n)
	Utils.logMessage("\nConvert to Laplacian Matrix finished")

	return D * mat * D

def computeEigenValsVectors(mat, dimensionReductionNum):
	eigenVals, eigenVecs = arpack.eigs(mat, k = dimensionReductionNum, tol=0, which = "LM")

	Utils.logMessage("\nCompute eigen values vectors finished")

	return eigenVals, eigenVecs


def unification(vecs):
	sq_sum = sqrt(multiply(vecs, vecs).sum(1))
	rows,cols = shape(vecs)

	for i in xrange(rows):
		for j in xrange(cols):
			vecs[i,j] = vecs[i,j]/sq_sum[i]

	Utils.logMessage("\nUnification finished")
	return vecs


def kMeans(vecs, clusterNum):
	clusters = KMeans.train(vecs, clusterNum, maxIterations=10, runs=10, initializationMode="random")

	Utils.logMessage("\nKmean cluster finished")

	return clusters

