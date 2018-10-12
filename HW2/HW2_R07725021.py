import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#import nltk
#nltk.download('punkt') #do this once

#initialnize
dict = {}

# 讀取document collection
for i in range(1,1096):
    file = open("IRTM/"+str(i)+".txt","r")
    # file to string array
    data = file.read()
    file.close()

    #tokenize
    tokendata = word_tokenize(data)
    regex = [c for c in tokendata if c.isalpha()] #去除non-alphanumeric
    #splitString = re.split('\W+',data) #會把10,000切成10 000

    #lowercasing
    lowerString = [elements.lower() for elements in regex]

    #stemming
    stemmer = PorterStemmer()
    stemString = [stemmer.stem(lowerstring) for lowerstring in lowerString]
    
    #stopword removal
    url = "http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words" 
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text,'html.parser') 
    stopwordList = soup.text # 抓取stopword list
    removeStopword = [word for word in stemString if word not in stopwordList]

    #儲存到dictionary
    for word in removeStopword:
        if(dict.get(word)): #有key，value+1
            dict[word] += 1
        else: #否則在dictionary加上該key，並從value=1開始
            dict[word] = 1   
    #print(dict)

t_index_pre = 1
for key,value in sorted(dict.items()):
    print(t_index_pre,key,value)
    t_index_pre = t_index_pre+1
# 將terms次數印到txt檔上，格式為t_index、term、df，並且是排序過(由小到大)
with open('filtered_file.txt','w') as f:
    t_index = 1
    for key,value in sorted(dict.items()):
        f.write("%d %s %d\n" % (t_index,key,value))
        t_index = t_index+1
