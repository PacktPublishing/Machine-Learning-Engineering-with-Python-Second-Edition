from pyspark.sql import SparkSession

from pyspark.sql import functions as f
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.feature import StandardScaler, OneHotEncoderEstimator, StringIndexer, VectorAssembler

from pyspark.ml import Pipeline, PipelineModel

categoricalColumns = ["job", "marital", "education", "contact", "housing", "loan", "default", "day"]

stages = []

spark = SparkSession.builder.getOrCreate()
data = spark.read.csv("../../chapter1/stream-classifier/data/banks.csv")

for categoricalCol in categoricalColumns:
    # fill some nulls
    #data = data.na.fill({categoricalCol:'Unknown'})
    # category indexing with string indexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index").setHandleInvalid("keep") # keep is for unknown categories
    # Use onehotencoder to convert cat variables into binary sparseVectors
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol+"classVec"])
    # Add stages. These are not run here, will be run later
    stages += [stringIndexer, encoder]

numericCols = ["numCol1"]
assemblerInputs = [c+"classVec" for c in categoricalColumns]+numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features") # features contains everything, not hot encoded and numerical
stages += [assembler]

# ML MODEL
from pyspark.ml.classification import LogisticRegression

data_transformation_pipeline = Pipeline().setStages(stages).fit(data)
prepped_data = data_transformation_pipeline.transform(data)

# Random train test split with seed
(trainingData, testData) = prepped_data.randomSplit([0.7,0.3], seed=100)

# model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

lrModel = lr.fit(trainingData)

# evaluate
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

paramGrid = (ParamGridBuilder()
                .addGrid(lr.regParam, [0.01, 0.5, 2.0])
                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                .addGrid(lr.maxIter, [1, 5, 10])
                .build())
estimatorPipeline = Pipeline(stages=[lr])

cv = CrossValidator(
    estimator = estimatorPipeline,
    estimatorParamMaps = paramGrid,
    evaluator = evaluator,
    numFolds=5
)

cvModel = cv.fit(trainingData)


# MODEL EVALUATION
def evaluate(predictionAndLabels):
    log = {}
    # show validation score (AUROC)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
    log['AUROC'] = "%f" % evaluator.evaluate(predictionAndLabels)
    print("Area under ROC = {}".format(log['AUROC']))

    # show validation score (AUPR)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
    log["AUPR"] = "%f" % evaluator.evaluate(predictionAndLabels)
    print("Area under PR = {}".format(log["AUPR"]))

    # Metrics
    predictionRDD = predictionAndLabels.select(['label','prediction']).rdd.map(lambda line: (line[1], line[0]))
    metrics = MulticlassMetrics(predictionRDD)

    print(metrics.confusionMatrix().toArray())

    # Overall statistics
    log['precision'] = "%s" % metrics.precision()
    log['recall'] = "%s" % metrics.recall()
    log['F1_measure'] = "%s" % metrics.fMeasure()

    # Statistics by class
    labels =  [0.0, 1.0]
    for label in sorted(labels):
        log[label] = {}
        log[label]['precision'] = "%s" % metrics.precision(label)
        log[label]['recall'] = "%s" % metrics.recall()
        log[label]['F1_measure'] = "%s" % metrics.fMeasure(label, beta=1.0)

    return log







