#-*-coding:utf-8-*-
targetFields = ['order_id','buyer_pin','buyer_full_name','buyer_full_address','buyer_mobile','buyer_ip','equipment_id','buyer_city_name','buyer_country_name','buyer_poi','promotion_id']

simWeight = {'buyer_ip':20, 
			 'buyer_mobile':0, 
			 'buyer_full_address':0, 
			 'equipment_id': 30, 
			 'buyer_full_name':0,
			 'buyer_city_name':0,
			 'buyer_county_name':0,
			 'buyer_poi':30,
			 'promotion_id':20}

DEFAULTSIM = 0.0

sourceFile ='/home/yifei/TestData/data/realdata/testdata_20160307.csv' 
truncatedFile = '/home/yifei/TestData/data/realdata/testdata_truncated_20160307.csv'
processedFile = '/home/yifei/TestData/data/realdata/testdata_processed_20160307.csv'
trainingFile = processedFile
testFile = ''
eigenVecFile = '/home/yifei/TestData/data/realdata/testdata_eigenVec_20160307.csv'
clusterIDCenterFile = '/home/yifei/TestData/data/realdata/testdata_clustered_id_center_20160307.csv'
clusterIDFile = '/home/yifei/TestData/data/realdata/testdata_clustered_id_20160307.csv'
IDFOREACHCLUSTER = 20
truncateLineCount = 3001
clusterNum = truncateLineCount/IDFOREACHCLUSTER
dimensionReductionNum = 15
treeMaxDepth = 3
treeMaxBins = 8

#below parameters only for classification model
confirmedAccountFile = ''
otherAccountFile = '' 
