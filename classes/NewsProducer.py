import requests
from bs4 import BeautifulSoup
import re
import json
from kafka import KafkaProducer

class NewsProducer:
    def __init__(self, broker, topic):
        self.producer = KafkaProducer(
            bootstrap_servers=broker,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic

    def fetch_and_produce(self, url):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        script_tag = soup.find("script", text=re.compile(r"var listing ="))
        if not script_tag:
            return

        javascript_code = script_tag.string
        match = re.search(r"var listing = ({.*?});", javascript_code)
        if not match:
            return
        
        try:
            listing_data = json.loads(match.group(1))
            for article in listing_data["data"]:
                title = article["article_title"]
                link = "https://www.thestar.com.my/" + article["permalink"]

                publish_time = article["publish_time"]
                
                # Extract summary from article_custom_fields
                custom_fields_json = article["article_custom_fields"]
                custom_fields = json.loads(custom_fields_json)
                summary = custom_fields.get("summary", ["Summary not available."])[0]
                
                article_body = BeautifulSoup(article["article_body"], "html.parser").get_text(separator="\n").strip()
                location_match = re.match(r'^([A-Z\s]+):\s', article_body)
                location = location_match.group(1).strip() if location_match else "Location not found"
                section = article.get("section_name", "N/A")
                sub_section = article.get("sub_section_name", "N/A")
                keywords = article.get("keyword_tagging", [])

                article_data = {
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "publish_time": publish_time,
                    "section": section,
                    "sub_section": sub_section,
                    "keywords": keywords,
                    "location": location,
                    "body": article_body,
                }

                # Send article data to Kafka
                self.producer.send(self.topic, article_data)
        except json.JSONDecodeError:
            print("Error")
        
        self.producer.flush()
        
    def close(self):
        self.producer.close()