from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum, udf, col
from pyspark.sql.types import StringType

import time

spark = SparkSession.builder.appName('luigi').getOrCreate()
dfT = spark.read.csv('/data/ethereum/blocks', header=True).orderBy("timestamp") #.select('_c2', '_c3').withColumnRenamed("_c2", "_c0").groupBy("_c0").agg(_sum("_c3")) #.filter(col('Country Code') == 'ES').withColumnRenamed("Criteria ID", "City").select('City', 'Name')
# dfC = spark.read.csv('C.csv', header=False).select('_c0') #.filter(col('Country Code') == 'ES').withColumnRenamed("Criteria ID", "City").select('City', 'Name')
# dfT.show(10)

udfT = udf(lambda x: time.strftime("%Y %m", time.gmtime(x)), StringType()) #Define UDF function

dfT = dfT.withColumn('time', udfT(col('timestamp').cast("integer"))).groupBy("time").agg(_sum("transaction_count")).orderBy("time")
#dfT.show()
dfT.repartition(1).write.csv("outASpark", sep=",",header=True)