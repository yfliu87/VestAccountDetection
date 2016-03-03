targetFields = ['order_id','buyer_pin','buyer_full_name','buyer_full_address','buyer_mobile','buyer_ip','equipment_id','buyer_city_name','buyer_country_name','buyer_poi']

simWeight = {'buyer_ip':10, 
			 'buyer_mobile':10, 
			 'buyer_full_address':10, 
			 'equipment_id': 10, 
			 'buyer_full_name':10,
			 'buyer_city_name':10,
			 'buyer_county_name':20,
			 'buyer_poi':20}

DEFAULTSIM = 0.0