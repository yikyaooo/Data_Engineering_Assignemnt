from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, StringType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsLoader:
    def __init__(self, hdfs_path):
        self.hdfs_path = hdfs_path  # Store path, not Spark objects

    def load_data(self, spark):
        spark.sparkContext.setLogLevel("ERROR")  # Only show errors, no warnings
        df = spark.read.option("multiline", "true").json(self.hdfs_path)
        return df

    def add_sentiment_label(self, df, col):
        sia = SentimentIntensityAnalyzer()  # Move inside method

        sentiment_score_udf = udf(lambda text: float(sia.polarity_scores(text)["compound"]) if text else 0.0, DoubleType())
        sentiment_udf = udf(lambda score: "positive" if score >= 0.05 else ("negative" if score <= -0.05 else "neutral"), StringType())

        df = df.withColumn("sentiment_score", sentiment_score_udf(df[col]))
        df = df.withColumn("sentiment", sentiment_udf(df["sentiment_score"]))
        return df