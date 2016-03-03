from pyspark import SparkContext
import codecs

#spark rdd can' t handle chinese 

field_idx_map = {'order_id':2,
				'buyer_pin':27,
				'buyer_full_name':29,
				'buyer_full_address':30,
				'buyer_mobile':32,
				'buyer_ip':112,
				'equipment_id':113,
				'buyer_city_name':89,
				'buyer_county_name':90,
				'buyer_poi':114}

def main(sourceFile, targetFile):
	sc = SparkContext()
	rawRDD = sc.textFile(sourceFile, use_unicode=False)

	pin_addr_map = rawRDD.map(lambda item: getFields(item)).groupByKey().mapValues(list).collect()

	writer = codecs.open(targetFile, 'w','gbk')
	for item in pin_addr_map:
		writer.write(item[0])
		writer.write(',')
		writer.write(item[1])
		writer.write('\n')

	writer.close()

	reader = codecs.open(targetFile,'r','gbk')
	line = reader.readline()
	while line:
		user = line.split(',')[0].encode('utf-8')
		address = line.split(',')[1].encode('utf-8')
		print address
		result = []
		for add in address:
			result.append(add.encode('utf-8'))

		line = reader.readline()
		

	reader.close()



def getFields(item):
	fields = item.split(',')
	buyer_pin = fields[field_idx_map['buyer_pin']]
	buyer_city = fields[field_idx_map['buyer_city_name']]
	buyer_county = fields[field_idx_map['buyer_county_name']]
	buyer_poi = fields[field_idx_map['buyer_poi']]

	buyer_address = combine(buyer_city, buyer_county, buyer_poi)

	return (buyer_pin,buyer_address)


def combine(city, county, poi):
	ret = ''
	if not isinstance(city,float):
		ret += city.encode('gbk')
		print ret
	if not isinstance(county,float):
		ret += county
	if not isinstance(poi,float):
		ret += poi

	print ret.decode('utf-8')
	return ret

if __name__ == '__main__':
	sourceFile = '/home/yifei/TestData/data/order_raw_20160229.csv'
	targetFile = '/home/yifei/TestData/data/order_map_20160229.txt'
	main(sourceFile, targetFile)