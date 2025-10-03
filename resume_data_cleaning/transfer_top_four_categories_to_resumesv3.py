from utils.mongohelper import MongoHelper


db = MongoHelper("cvbuilder")

resumes = db.get_collection("resumesv2").find({"JobCategory.EnTitle": { "$in": ["Electrical and Electronic Engineering", "Civil Engineering / Architecture", "Software Development and Programming","Mechanical / Aerospace Engineering"]}})

db.get_collection("resumesv3").insert_many(resumes)