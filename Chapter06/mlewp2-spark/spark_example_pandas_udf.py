# run the following in the mleip-pd-udf environment
#Note - reuses some code from chapter 3
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import pandas_udf, PandasUDFType
@pandas_udf('double', PandasUDFType.SCALAR)
def pandas_plus_one(v):
    # `v` is a pandas Series
    return v.add(1)  # outputs a pandas Series
spark.range(10).select(pandas_plus_one("id")).show()


#
import sklearn.svm
import sklearn.datasets
clf = sklearn.svm.SVC()
X, y = sklearn.datasets.load_wine(return_X_y=True)
clf.fit(X, y)

df = spark.createDataFrame(X.tolist())

import pandas as pd
@pandas_udf(returnType=IntegerType())
def predict_pd_udf(*cols):
    X = pd.concat(cols, axis=1)
    return pd.Series(clf.predict(X))

col_names = ['_{}'.format(x) for x in range(1, 14)]

df_pred = df.select('*', predict_pd_udf(*col_names).alias('class'))
df_pred.show()







