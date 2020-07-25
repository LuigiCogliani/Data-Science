"""
PART B
we want to evaluate the top 10 smart contracts by total Ether received.
We will get the total Ether received from each address and check which address is also present in the
"contracts" dataset, hence which address is actually a contract. The we will sort and get the top 10 results.


"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum

spark = SparkSession.builder.appName('luigi').getOrCreate()

"""
We can now work with dataframes. 

This line will:
a. read the "transactions" csv
b. get the columns to_address and value
c. rename to_address into address
d. group by address
e. sum over value

the result will be a dataframe with two column, address and value. Each entry will be a unique address with the total 
value of Ether received.
"""
dfT = spark.read.csv('/data/ethereum/transactions', header=True).select('to_address', 'value').withColumnRenamed("to_address", "address").groupBy("address").agg(_sum("value"))

"""
read the "contracts" csv and get only the address column
"""
dfC = spark.read.csv('/data/ethereum/contracts', header=True).select('address')

"""
We can now join the two datasets over the column address: this will filter out the rows in dfT containing an address 
not present in the dfC dataframe 
"""
inner_join = dfT.join(dfC, "address")

#sort over sum of values in ascending order
order = inner_join.sort('sum(value)', ascending=False)
"""
get the top 10 values. It is worth nothing that .head returns a list, hence we cannot treat top10 as a dataframe

"""
top10=order.head(10)


#finally, we can print our results.
for record in top10:
    print("{};{}".format(record[0],record[1]))

