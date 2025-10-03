import pymongo


class MongoHelper:
    def __init__(self, db_name, server="mongodb+srv://aminslkbgh:yL2xRioU6S0rChkw@resumescreening.0tmrgjg.mongodb.net/?retryWrites=true&w=majority&appName=resumescreening"):
        myclient = pymongo.MongoClient(server)
        self.db = myclient[db_name]

    def find_documents(self, collection_name, query,projection):
        collection = self.db[collection_name]
        return collection.find(query,projection)

    def get_collection(self,collection_name):
        return self.db[collection_name]
