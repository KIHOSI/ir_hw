import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#import nltk
#nltk.download('punkt') #do this once

#initialnize
dict_all = {}
dict_tf = {}

# 讀取document collection
for i in range(1,1096):
    file = open("IRTM/"+str(i)+".txt","r")
    # file to string array
    data = file.read()
    file.close()

    #tokenize(改用nltk的tokenizer)
    tokendata = word_tokenize(data)
    regex = [c for c in tokendata if c.isalpha()] #去除non-alphanumeric

    #lowercasing
    lowerString = [elements.lower() for elements in regex]

    #stemming
    stemmer = PorterStemmer()
    stemString = [stemmer.stem(lowerstring) for lowerstring in lowerString]
    
    #stopword removal
    #因為會有回應時間過長問題，直接抓下來讀取檔案
    file2 = open("stopwordlist.txt","r")
    stopwordList = file2.read()
    file2.close()
    removeStopword = [word for word in stemString if word not in stopwordList]

    #儲存到dictionary
    dict_now = {} #儲存該doc的dict
    for word in removeStopword:
        if(dict_now.get(word)): #有key，value+1
            dict_now[word] += 1
        else: #否則在dictionary加上該key，並從value=1開始
            dict_now[word] = 1 

    #將每一個doc的term的tf先輸出成txt(ex. 1.txt...)
    with open('TermFrequency/'+str(i)+'.txt','w+') as f: #w+，沒有存在該檔案，則create(但Folder一定要存在)
        for key,value in sorted(dict_now.items()):
            f.write("%s %d\n" % (key,value))        

    #用dict_now儲存全部，dict_now儲存每一個doc的
    #判斷dict_now是否已經存有該term
    for key,value in dict_now.items():
        if(dict_all.get(key)):
            dict_all[key] += 1
        else:
            dict_all[key] = 1           

# 將terms次數印到txt檔上，格式為t_index、term、df，並且是排序過(依據字母由小到大)
with open('dictionary.txt','w') as f:
    t_index = 1
    for key,value in sorted(dict_all.items()):
        f.write("%d %s %d\n" % (t_index,key,value))
        t_index = t_index+1
