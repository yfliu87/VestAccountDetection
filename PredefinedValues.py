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