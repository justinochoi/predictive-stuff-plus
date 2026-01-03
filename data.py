import pybaseball as pb 
from pybaseball import cache 

# just for personal purposes 
cache.enable() 

statcast20 = pb.statcast(start_dt='2020-07-23', end_dt='2020-09-27')
statcast21 = pb.statcast(start_dt='2021-04-01', end_dt='2021-10-03')
statcast22 = pb.statcast(start_dt='2022-04-07', end_dt='2022-10-05')
statcast23 = pb.statcast(start_dt='2023-03-30', end_dt='2023-10-01')

statcast20.to_parquet("/Users/justinchoi/BaseballData/statcast_20.parquet") 
statcast21.to_parquet("/Users/justinchoi/BaseballData/statcast_21.parquet") 
statcast22.to_parquet("/Users/justinchoi/BaseballData/statcast_22.parquet") 
statcast23.to_parquet("/Users/justinchoi/BaseballData/statcast_23.parquet") 
