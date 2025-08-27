import subprocess
import json
from kafka import KafkaConsumer, TopicPartition

class NewsConsumer:
    def __init__(self, broker, topic, group_id, local_path, hdfs_path):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=broker,
            auto_offset_reset='earliest',
            enable_auto_commit=False,  # Manually commit offsets
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.topic = topic
        self.local_path = local_path
        self.hdfs_path = hdfs_path  # HDFS directory

    def save_to_hdfs(self, articles):
        """Save JSON articles to HDFS."""

        # Save data locally
        try:
            with open(self.local_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving to local file: {e}")
            return
        
        # Upload to HDFS using subprocess
        command = ["hdfs", "dfs", "-put", "-f", self.local_path, self.hdfs_path]
        try:
            subprocess.run(command, check=True)
            print("Data successfully saved to HDFS.")
        except subprocess.CalledProcessError as e:
            print(f"Error uploading to HDFS: {e}")

    def consume_articles(self):
        """Consumes articles from Kafka and stores them in a list."""
        
        partitions = self.consumer.partitions_for_topic(self.topic)
        topic_partitions = [TopicPartition(self.topic, p) for p in partitions]

        # Get end offsets
        end_offsets = self.consumer.end_offsets(topic_partitions)
        
        articles = []  # Store articles before writing to HDFS
        
        for message in self.consumer:
            article = message.value
            print("\nConsumed Article:")
            print(json.dumps(article, indent=4))
            
            articles.append(article)

            # Stop when all messages have been consumed
            if all(self.consumer.position(tp) >= end_offsets[tp] for tp in topic_partitions):
                break

        self.consumer.close()
        return articles  # Return articles for further processing