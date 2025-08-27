from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql.functions import lower, trim, regexp_replace
import joblib

class TextPreprocessing():

    def labelEncode(self, df, inputName, outputName):
        label_indexer = StringIndexer(inputCol= inputName, outputCol=outputName).fit(df)
        df = label_indexer.transform(df)
        return df

    def remove_duplicate(self, df):
        df = df.dropDuplicates()
        return df

    def lowercase(self, df, x):
        df = df.withColumn(x, lower(trim(df[x])))
        return df

    def remove_special(self, df, x):
        df = df.withColumn(x, regexp_replace(df[x], "[^a-zA-Z\s]", ""))  # Remove special characters
        return df

    def tokenize(self, df, inputName, outputName):
        tokenizer = Tokenizer(inputCol=inputName, outputCol=outputName)
        df = tokenizer.transform(df)
        return df

    def removeStopwords(self, df, inputName, outputName):
        stopword_remover = StopWordsRemover(inputCol=inputName, outputCol=outputName)
        df = stopword_remover.transform(df)
        return df

    def hashingTF(self, df, inputName, outputName):
        hashing_tf = HashingTF(inputCol=inputName, outputCol=outputName)
        df = hashing_tf.transform(df)
        return df

    def idf(self, df, inputName, outputName):
        idf = IDF(inputCol=inputName, outputCol=outputName)
        df = idf.fit(df).transform(df)
        return df

    def fitPipeline(self, df, x, y):
        label_indexer = StringIndexer(inputCol= y, outputCol="indexedLabel").fit(df)
        tokenizer = Tokenizer(inputCol=x, outputCol="words")
        stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features")
        idf = IDF(inputCol="raw_features", outputCol="features")
        
        pipeline = Pipeline(stages=[label_indexer, tokenizer, stopwords_remover, hashing_tf, idf])
        model = pipeline.fit(df)
        return model

    def save_pipeline_as_pkl(self, model, file_name):
        try:
            joblib.dump(model, file_name)
            print(f"Model successfully !")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def save_pipeline_to_hdfs(self, model, file_name, hdfs_path):
        model_path = f"{hdfs_path}/{file_name}"  
        
        try:
            model.write().overwrite().save(model_path)
            print(f"Model successfully saved at: {model_path}")
        except Exception as e:
            print(f"Failed to save model locally: {e}")
