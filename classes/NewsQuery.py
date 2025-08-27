from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

class NewsQuery:
    def __init__(self, collection):
        self.collection = collection

    def count_location_positive(self, location):
        query = {
            "predicted_label": "positive",
            "$or": [
                {"title": {"$regex": location, "$options": "i"}},
                {"summary": {"$regex": location, "$options": "i"}}
            ]
        }
        return self.collection.count_documents(query)

    def get_recent_by_location_and_label(self, n = 5):
        query = {
            "location": {"$exists": True, "$ne": ""},
            "predicted_label": {"$in": ["negative", "neutral"]}
        }
        return list(self.collection.find(query).sort("publish_time", -1).limit(n))

    def count_empty_keywords_by_sub_section(self):
        pipeline = [
            {"$match": {"keywords": []}},
            {"$group": {
                "_id": "$sub_section",
                "empty_keywords_count": {"$sum": 1}
            }},
            {"$sort": {"empty_keywords_count": -1}}
        ]
        return list(self.collection.aggregate(pipeline))

    def average_articles_per_day(self, start_date: datetime, end_date: datetime):
        pipeline = [
            {"$addFields": {"publish_date": {"$toDate": "$publish_time"}}},
            {"$match": {"publish_date": {"$gte": start_date, "$lt": end_date}}},
            {"$group": {
                "_id": {
                    "year": {"$year": "$publish_date"},
                    "month": {"$month": "$publish_date"},
                    "day": {"$dayOfMonth": "$publish_date"}
                },
                "count": {"$sum": 1}
            }},
            {"$group": {"_id": None, "average_articles_per_day": {"$avg": "$count"}}}
        ]
        return list(self.collection.aggregate(pipeline))

    def group_by_label_latest(self):
        pipeline = [
            {"$group": {
                "_id": "$predicted_label",
                "count": {"$sum": 1},
                "latest_article": {"$max": "$publish_time"}
            }},
            {"$sort": {"count": -1}}
        ]
        return list(self.collection.aggregate(pipeline))

    def top_locations_with_positive(self, limit=3):
        pipeline = [
            {"$match": {"predicted_label": "positive", "location": {"$ne": ""}}},
            {"$group": {"_id": "$location", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        return list(self.collection.aggregate(pipeline))
        

    def search_articles_with_keywords(self, keywords: list):
        regex = "(?=.*" + ")(?=.*".join(keywords) + ")"
        query = {
            "$and": [
                {
                    "$or": [
                        {"title": {"$regex": regex, "$options": "i"}},
                        {"summary": {"$regex": regex, "$options": "i"}}
                    ]
                }
            ]
        }
        return list(self.collection.find(query))

        

    def monthly_sentiment_distribution(self):
        pipeline = [
            {"$addFields": {"publish_date": {"$toDate": "$publish_time"}}},
            {"$group": {
                "_id": {
                    "year": {"$year": "$publish_date"},
                    "month": {"$month": "$publish_date"},
                    "sentiment": "$predicted_label"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1, "count": -1}}
        ]
        return list(self.collection.aggregate(pipeline))

    def location_sentiment_heatmap(self):
        pipeline = [
            {"$match": {"location": {"$ne": ""}}},
            {"$group": {
                "_id": {"location": "$location", "label": "$predicted_label"},
                "count": {"$sum": 1}
            }},
            {"$group": {
                "_id": "$_id.location",
                "total": {"$sum": "$count"},
                "sentiments": {"$push": {"label": "$_id.label", "count": "$count"}}
            }},
            {"$sort": {"total": -1}}
        ]
        return list(self.collection.aggregate(pipeline))

    def publishing_peak_hours(self):
        pipeline = [
            {"$addFields": {"publish_date": {"$toDate": "$publish_time"}}},
            {"$project": {"hour": {"$hour": "$publish_date"}}},
            {"$group": {"_id": "$hour", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        return list(self.collection.aggregate(pipeline))

    def low_content_quality_articles(self):
        pipeline = [
            {"$project": {
                "title": 1,
                "summary": 1,
                "summary_length": {"$strLenCP": "$summary"},
                "has_keywords": {
                    "$cond": [{"$gt": [{"$size": "$keywords"}, 0]}, True, False]
                }
            }},
            {"$match": {
                "$or": [
                    {"summary_length": {"$lt": 50}},
                    {"has_keywords": False}
                ]
            }},
            {"$sort": {"summary_length": 1}}
        ]
        return list(self.collection.aggregate(pipeline))