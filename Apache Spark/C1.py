from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum, udf, col, mean as _mean
from pyspark.sql.types import StringType

import time

spark = SparkSession.builder.appName('luigi').getOrCreate()
# avoid crating a new spark session everytime the code is executed


dfG = spark.read.csv('/data/ethereum/transactions', header=True).select('block_timestamp', 'gas').orderBy("block_timestamp")
# take the csv without the header, rename columns 2 ("to_address") and 3 ("value") renaming them c2 and c0, group over c0 summing c3

udfG = udf(lambda x: time.strftime("%Y %m", time.gmtime(x)), StringType()) #Define UDF function

dfG = dfG.withColumn('time', udfG(col('block_timestamp').cast("integer"))).groupBy("time").agg(_mean("gas")).orderBy("time")
#dfG.show()
dfG.repartition(1).write.csv("outCSpark", sep=",",header=True)


