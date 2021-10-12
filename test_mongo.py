import pymongo
from pymongo import MongoClient
import ssl

if __name__ == '__main__':
    cluster = MongoClient("mongodb+srv://moran:Weglp887yt@cluster0.qeje5.mongodb.net/Wearables?retryWrites=true&w=majority",
                          connect=False, ssl=True, ssl_cert_reqs=ssl.CERT_NONE)

    db = cluster["Wearables"]

    collection = db["Watches"]

    #feilds in each post
    post = {"_id":118, "name":"Avital"}
    #inserting first entry = post.
    collection.insert_one(post)
    a=1
