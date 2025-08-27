from pyspark.sql.functions import col, from_json, explode, window, count, lower, date_format
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

class NewsStreaming:
    def read_from_kafka(self, spark, kafka_broker, kafka_topic):
        """Read streaming data from Kafka and extract JSON string"""
        return spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("subscribe", kafka_topic) \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .load() \
            .selectExpr("CAST(value AS STRING) as json_data")

    def process_stream(self, df, schema: StructType):
        """Process Kafka stream by parsing JSON and extracting fields"""
        # Parse the JSON in the 'json_data' column using the provided schema
        parsed_df = df.withColumn("data", from_json(col("json_data"), schema)) \
                      .select("data.*")  # Unpack the JSON object into columns
        return parsed_df

    def preview_stream(self, df, num_rows=20, truncate=False):
        """Preview streaming DataFrame to console (for debugging)"""
        return df.writeStream \
            .format("console") \
            .outputMode("append") \
            .option("truncate", str(truncate).lower()) \
            .option("numRows", str(num_rows)) \
            .start()

        
    def analyse_stream(self, df, model):
        df = model.transform(df)
        return df

    def filter_by_sub_section(self, df, section_name):
        return df.filter(lower(col("sub_section")) == section_name.lower())

    def filter_by_sentiment(self, df, sentiment_value):
        return df.filter(lower(col("predicted_label")) == sentiment_value.lower())

    def filter_by_date(self, df, start_date, end_date):
        return df.filter((col("publish_time") >= start_date) & (col("publish_time") <= end_date))

    def select_columns(self, df, columns):
        return df.select(*columns)

    def aggregate_by_location(self, df):
        return df.groupBy("location").agg(count("*").alias("news_count"))

    def aggregate_by_section(self, df):
        return df.groupBy("section").agg(count("*").alias("news_count"))

    def explode_keywords(self, df):
        return df.withColumn("keyword", explode("keywords"))

    def keyword_count(self, df):
        exploded = self.explode_keywords(df)
        return exploded.groupBy("keyword").agg(count("*").alias("keyword_freq"))

    def windowed_sentiment_count(self, df, window_duration="10 minutes", slide_duration="5 minutes"):
        return df \
            .withWatermark("publish_time", "1 hour") \
            .groupBy(window(col("publish_time"), window_duration, slide_duration), col("predicted_label")) \
            .count()


    def write_output_console(self, df, trigger_interval=None, output_mode="append"):
        writer = df.writeStream.outputMode(output_mode)
        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)
            
        return writer.format("console").start()

    def write_output_memory(self, df, queryName, trigger_interval=None, output_mode="append"):
        writer = df.writeStream.outputMode(output_mode)
        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)
            
        return writer.format("memory").queryName(queryName).start()

    def write_output_hdfs(self, df, hdfs_path, trigger_interval=None):
        if trigger_interval:
            writer = writer.trigger(processingTime=trigger_interval)
            
        return writer.format("parquet") \
            .option("path", hdfs_path) \
            .option("checkpointLocation", f"{hdfs_path}/checkpoint") \
            .start()



