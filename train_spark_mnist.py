from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("mnist_test").getOrCreate()
sc = spark.sparkContext


minst_df_with_id = spark.read.parquet('mnist_data_parquet')



from pyspark.ml.feature import VectorAssembler
input_cols= minst_df_with_id.columns[2:-1]

assembler = VectorAssembler(inputCols = input_cols, outputCol = 'features')

# assembler.transform(minst_df_with_id).show()

# pipeline = Pipeline(stages=[assembler])
# model = pipeline.fit(mnist_df)



from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier


classifier = RandomForestClassifier(labelCol = 'target', featuresCol = 'features', numTrees= 10)
#GB_classifier = GBTClassifier(labelCol = 'target', featuresCol = 'features', maxIter=10)

pipeline = Pipeline(stages=[assembler, classifier])

(train, test) = minst_df_with_id.select(minst_df_with_id.columns[1:]).randomSplit([0.7, 0.3])

model = pipeline.fit(train)
model.save()


