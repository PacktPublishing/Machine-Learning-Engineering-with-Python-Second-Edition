#Note - reuses some code from chapter 3
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import StringType
from pyspark.sql import functions as f


# Create spark context
sc = SparkContext("local", "Ch6BasicExampleApp")
# Get spark session
spark = SparkSession.builder.getOrCreate()

# Get the data and place it in a spark dataframe
#data = spark.read.format("csv").option("sep", ";").option("inferSchema", "true").option("header", "true").load(
#    "../chapter1/stream-classifier/data/bank/bank.csv")
data = spark.read.format("csv").option("sep", ";").option("inferSchema", "true").option("header", "true").load(
   "data/bank/bank.csv")

# map target to numerical category
#data = data.withColumn('label', f.when((f.col("y") == "yes"), 1).otherwise(0))

data.printSchema()

data.show()

# UDF
import datetime
def month_as_int(month):
    month_number = datetime.datetime.strptime(month, "%b").month
    return month_number

spark.udf.register("monthAsInt", month_as_int, StringType())


# Apply in spark sql
data.createOrReplaceTempView('bank_data_view')

spark.sql('''
select *, monthAsInt(month) as month_as_int from bank_data_view
''').show()


# Apply on dataframe
from pyspark.sql.functions import udf
month_as_int_udf = udf(month_as_int, StringType())

df = spark.table("bank_data_view")
df.withColumn('month_as_int', month_as_int_udf("month")).show()

# Create with decorator syntax
@udf("string")
def month_as_int_udf(month):
    month_number = datetime.datetime.strptime(month, "%b").month
    return month_number

df.withColumn('month_as_int', month_as_int_udf("month")).show()


from pyspark.sql.functions import pandas_udf, PandasUDFType

# @pandas_udf('string')
# def month_as_int(month_series):
#     return datetime.datetime.strptime()