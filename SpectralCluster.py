from __future__ import division
import scipy.io as scio
import Utils
from scipy import sparse
from scipy.sparse.linalg.eigen import arpack
from numpy import *
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext


def getClusters(mat, rawdata, outputFilePath,num_clusters):
	sc = SparkContext()

	laplacianMat = getLaplacianMatrix(mat)

	vals, vecs = computeEigenValsVectors(laplacianMat, num_clusters)

	unifiedRDDVecs = sc.parallelize(unification(vecs))

	model = kMeans(unifiedRDDVecs,num_clusters)

	Utils.logMessage("\nSpectral cluster finished") 

	return model, unifiedRDDVecs


def getLaplacianMatrix(mat):
	D = mat.sum(1)
	D = sqrt(1/D)
	n = len(D)
	D = D.T
	D = sparse.spdiags(D, 0, n, n)
	Utils.logMessage("\nConvert to Laplacian Matrix finished")

	return D * mat * D

def computeEigenValsVectors(mat, num_clusters):
	eigenVals, eigenVecs = arpack.eigs(mat, k = num_clusters, tol=0, which = "LM")

	Utils.logMessage("\nCompute eigen values vectors  finished")

	return eigenVals, eigenVecs


def unification(vecs):
	sq_sum = sqrt(multiply(vecs, vecs).sum(1))
	rows,cols = shape(vecs)

	for i in xrange(rows):
		for j in xrange(cols):
			vecs[i,j] = vecs[i,j]/sq_sum[i]

	Utils.logMessage("\nUnification finished")
	return vecs


def kMeans(vecs, num_clusters):
	clusters = KMeans.train(vecs, num_clusters, maxIterations=10, runs=10, initializationMode="random")

	Utils.logMessage("\nKmean cluster finished")

	return clusters

