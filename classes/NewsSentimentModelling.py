from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression,  LinearSVC, OneVsRest
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import joblib

class NewsSentimentModelling:
    def __init__(self, df, x, y, train_size):
        self.x = x
        self.y = y
        self.df = df.select(x, y).dropna()
        self.train_data, self.test_data = self.df.randomSplit([train_size, 1-train_size], seed=42)
    
        self.accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol= self.y, predictionCol="prediction", metricName="accuracy"
        )
    
        self.precision_evaluator = MulticlassClassificationEvaluator(
            labelCol= self.y, predictionCol="prediction", metricName="weightedPrecision"
        )
        
        self.recall_evaluator = MulticlassClassificationEvaluator(
            labelCol= self.y, predictionCol="prediction", metricName="weightedRecall"
        )
        
        self.f1_evaluator = MulticlassClassificationEvaluator(
            labelCol= self.y, predictionCol="prediction", metricName="f1"
        )

    def train_naive_bayes(self):
        model = NaiveBayes(labelCol=self.y, featuresCol= self.x)
        param_grid = ParamGridBuilder().addGrid(model.smoothing, [0.0, 1.0]).build()
        return self.cross_validate(model, param_grid)

    def train_random_forest(self):
        model = RandomForestClassifier(labelCol=self.y, featuresCol= self.x)
        param_grid = ParamGridBuilder()
        param_grid = param_grid.addGrid(model.numTrees, [5, 10])
        param_grid = param_grid.addGrid(model.maxDepth, [3, 5])
        param_grid = param_grid.addGrid(model.maxBins, [8, 16]).build()
        return self.cross_validate(model, param_grid)

    def train_decision_tree(self):
        model = DecisionTreeClassifier(labelCol=self.y, featuresCol= self.x)
        param_grid = ParamGridBuilder()
        param_grid = param_grid.addGrid(model.maxDepth, [3, 5])
        param_grid = param_grid.addGrid(model.maxBins, [8, 16]).build()
        return self.cross_validate(model, param_grid)

    def train_linear_svm(self, maxIter = 10):
        base_svm = LinearSVC(labelCol=self.y, featuresCol=self.x, maxIter=maxIter)
    
        model = OneVsRest(classifier=base_svm, labelCol=self.y, featuresCol=self.x)
        
        param_grid = ParamGridBuilder() \
            .addGrid(base_svm.regParam, [0.1, 0.01]) \
            .build()
    
        return self.cross_validate(model, param_grid)

    def train_logistic_regression(self, maxIter = 10):
        model = LogisticRegression(labelCol=self.y, featuresCol=self.x, maxIter=maxIter)
        
        param_grid = ParamGridBuilder() \
            .addGrid(model.regParam, [0.0, 0.1]) \
            .addGrid(model.elasticNetParam, [0.0, 0.5]) \
            .build()
        
        return self.cross_validate(model, param_grid)


    def cross_validate(self, model, param_grid):
        crossval = CrossValidator(
            estimator=model, 
            estimatorParamMaps=param_grid, 
            evaluator=self.accuracy_evaluator,
            numFolds=3
        )
        return crossval.fit(self.train_data)

    def evaluate_model(self, model):
        predictions = model.transform(self.test_data)
        
        accuracy = self.accuracy_evaluator.evaluate(predictions)
        precision = self.precision_evaluator.evaluate(predictions)
        recall = self.recall_evaluator.evaluate(predictions)
        f1_score = self.f1_evaluator.evaluate(predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")
        
        return [accuracy, precision, recall, f1_score]

    def save_model_as_pkl(self, model, file_name):
        try:
            joblib.dump(model, file_name)
            print("Model successfully saved!")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def save_model_to_hdfs(self, model, file_name, hdfs_path):
        model_path = f"{hdfs_path}/{file_name}"  
        
        try:
            model.write().overwrite().save(model_path)
            print(f"Model successfully saved at: {model_path}")
        except Exception as e:
            print(f"Failed to save model locally: {e}")
