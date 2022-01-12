import tweepy
# from rnnkeras import generate_sentence
# import tensorflow as tf
import random 
# from apscheduler.schedulers.blocking import BlockingScheduler





# Authenticate to Twitter
auth = tweepy.OAuthHandler("3DdjN76vPrqLPCQ55lwTSQQ0p", "qOHwkoI1T4Qnqe7hHDEVeqn0gVP2GtGcUxyxzsu0d4gNOedghC")
auth.set_access_token("1480287330502070273-Hp9GxTFfWccQTe2xfUS3CY1F4CTfVv", "Q7mpq1K5JgPr6QwUE8QKvrpJOA9WVKWrQ5m48OJbCWJEl")

# Create API object
api = tweepy.API(auth)
years = ["2021", "2022"]
months = ["01", "02","03","04","05","06","07","08","09","10","11"]
days = ["01", "02","03","04","05","06","07","08","09","10","11", "12","13","14","15","16","17","18","19","20","21", "22","23","24","25","26","27","28","29"]
sentence = []
for year in years:
    
    for month in months:
        
        for idx, day in enumerate(days):
            print(year)
            print(month)
            print(day)
          
            if idx != days[-1]:
                for tweet in api.search_full_archive("bot",query="freebritney", fromDate=year+month+day+"1200", toDate=year+month+days[idx+1]+"1200", maxResults=100 ):
                    print(f"{tweet.user.name}:{tweet.text}")
                    sentence.append(tweet.text)



   
print(sentence)
