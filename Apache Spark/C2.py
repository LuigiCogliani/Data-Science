from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum

spark = SparkSession.builder.appName('luigi').getOrCreate()
# avoid crating a new spark session everytime the code is executed



dfT = spark.read.csv('/data/ethereum/transactions', header=True).select('to_address', 'value').groupBy("to_address").agg(_sum("value"))
dfG = spark.read.csv('/data/ethereum/transactions', header=True).select('to_address', 'gas').groupBy("to_address").agg(_sum("gas"))
"""
top3=dfT.head(3)
df3 = spark.createDataFrame(top3)
df3.show()
"""


dfC = spark.read.csv('/data/ethereum/contracts', header=True).select('address').withColumnRenamed("address", "to_address")

inner_join = dfT.join(dfC, "to_address")
# join su _c0 e print
orderG = dfG.sort('sum(gas)', ascending=False)
top10C=orderG.head(10)

order = inner_join.sort('sum(value)', ascending=False)
top10B=order.head(10)
print('Top10 address per ETH received')
for record in top10B:
    print("{};{}".format(record[0],record[1]))
print('Top 10 Gas')
for record in top10C:
    print("{};{}".format(record[0],record[1]))
B = spark.createDataFrame(top10B)
C = spark.createDataFrame(top10C)
inner_j = B.join(C, "to_address")
print('address')
inner_j.show()