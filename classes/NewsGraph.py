from neo4j import GraphDatabase

class NewsGraph:
    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password
        self.auth = (username, password)
        self.driver = GraphDatabase.driver(uri, auth=self.auth)

    def verify_connection(self):
        try:
            self.driver.verify_connectivity()
            print("✅ Connection successful!")
        except Exception as e:
            print("❌ Connection failed:", e)

    def insert_spark_dataframe(self, spark_df, query):
        try:
            with self.driver.session() as session:
                for row in spark_df.collect():  # Caution: collect() brings all data to driver
                    session.write_transaction(self._insert_row_from_spark, query, row.asDict())
            print("Successfully inserted!")
        except Exception as e:
            print("Error inserting:", e)

    def _insert_row_from_spark(self, tx, query, row_dict):
        tx.run(query,
               title=row_dict["title"],
               summary=row_dict["summary"],
               publish_time=row_dict["publish_time_ts"],
               predicted_label=row_dict["predicted_label"],
               section=row_dict["section"],
               sub_section=row_dict["sub_section"],
               location=row_dict["location"])



    def run_cypher_query(self, spark, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            records = [record.data() for record in result]
    
        if not records:
            return spark.createDataFrame([], schema=[])
    
        return spark.createDataFrame(records)


    def close(self):
        self.driver.close()
