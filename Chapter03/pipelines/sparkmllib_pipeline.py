from pyspark.sql import SparkSession
from pyspark import SparkContext

from pyspark.sql import functions as f
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer, Imputer, VectorAssembler

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression

if __name__ == "__main__":
    # Create spark context
    sc = SparkContext("local", "pipelines")
    # Get spark session
    spark = SparkSession.builder.getOrCreate()

    # Get the data and place it in a spark dataframe
    data = spark.read.format("csv").option("sep", ";").option("inferSchema", "true").option("header", "true").load(
        "../../chapter1/stream-classifier/data/bank/bank.csv")

    # map target to numerical category
    data = data.withColumn('label', f.when((f.col("y") == "yes"), 1).otherwise(0))

    # define list for storage stage references
    stages = []

    # define the transformation stages for the categorical columns
    categoricalColumns = ["job", "marital", "education", "contact", "housing", "loan", "default", "day"]
    for categoricalCol in categoricalColumns:
        # fill some nulls
        # data = data.na.fill({categoricalCol:’Unknown’})
        # category indexing with string indexer
        stringIndexer = StringIndexer(inputCol=categoricalCol,
                                      outputCol=categoricalCol + "Index").setHandleInvalid(
            "keep")  # keep is for unknown categories
        # Use onehotencoder to convert cat variables into binary sparseVectors
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        # Add stages. These are not run here, will be run later
        stages += [stringIndexer, encoder]

    # define impute stage for the numerical columns
    numericalColumns = ["age", "balance"]
    numericalColumnsImputed = [x + "_imputed" for x in numericalColumns]
    imputer = Imputer(inputCols=numericalColumns, outputCols=numericalColumnsImputed)
    stages += [imputer]

    # define numerical assembler first for scaling
    numericalAssembler = VectorAssembler(inputCols=numericalColumnsImputed, outputCol='numerical_cols_imputed')
    stages += [numericalAssembler]

    # define the standard scaler stage for the numerical columns
    scaler = StandardScaler(inputCol='numerical_cols_imputed', outputCol="numerical_cols_imputed_scaled")
    stages += [scaler] # already a list so no need for brackets

    # Perform assembly stage to bring together features
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + ["numerical_cols_imputed_scaled"]
    # features contains everything, one hot encoded and numerical
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    # define the model stage at the end of the pipeline
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
    stages += [lr]

    # Random train test split with seed
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=100)

    # Define the entire pipeline and fit on the train data and transform on the test data
    clfPipeline = Pipeline().setStages(stages).fit(trainingData)
    clfPipeline.transform(testData)
