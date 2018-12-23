import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import math
from collections import defaultdict

class CountSimilarity():
    global dict_tf_idf 
    dict_tf_idf = defaultdict(dict)  #key為docid,value為term和sqrt_tfidf (dict中dict)

    def main(self): #主要運作流程，計算出tf-idf
        dict_df_all,dict_tf_each_doc = self.tokenize() #dict_tf_all:得到1095個doc所有term的tf。dict_tf_each_now:得每個一doc中各別的term和tf
        for i in range(1,1096): #計算tf-idf
            vectorLength = self.countDocVectorLength(dict_tf_each_doc[i]) #計算每一個doc的vectorlength
            for term,tf in dict_tf_each_doc[i].items():
                tf_idf = self.findIDF(dict_df_all,term,tf) #count tf-idf
                sqrt_tfidf = float(tf_idf/vectorLength) #get tf-idf unit vector
                dict_tf_idf[i][term] = sqrt_tfidf

    def tokenize(self):
        #initialnize
        dict_df_all = {}
        dict_tf_each_doc = defaultdict(dict)

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
            dict_tf_now = {} #儲存該doc的tf
            for word in removeStopword:
                if(dict_tf_now.get(word)): #有key，value+1
                    dict_tf_now[word] += 1
                else: #否則在dictionary加上該key，並從value=1開始
                    dict_tf_now[word] = 1 

            #儲存到dict_tf_each_doc，紀錄每個doc中term的tf。key為doc,value為term和他的tf(dict中dict)
            for term,tf in dict_tf_now.items():
                dict_tf_each_doc[i][term] = tf

            #用dict_all儲存全部term的tf，dict_now儲存每一個doc的tf
            #判斷dict_now是否已經存有該term
            for term in dict_tf_now.keys():
                if(dict_df_all.get(term)):
                    dict_df_all[term] += 1
                else:
                    dict_df_all[term] = 1           

        return dict_df_all,dict_tf_each_doc

    def countDocVectorLength(self,dict): #計算vector length
        squareNum = 0
        for term,tf in dict.items():
            squareNum += int(tf) * int(tf) #tf^2相加
        sqrtNum = math.sqrt(squareNum) #最後squart roots
        return sqrtNum  

    def findIDF(self,dict_df,term,tf): #該term的df，，將df變idf    
        #在dictionary.txt找到該term的df，將df變idf
        df = dict_df.get(term)
        idf = math.log10(1095/float(df))
        tf_idf = float(tf)*float(idf)
        return float(tf_idf) 

    def cosineSimilarity(self,dict_tf_idf,dict1,dict2): #計算兩篇文章的cosineSimilarity
        count = 0
        for term_1 in dict1: #比較dict2是否有包含dict1的key，有表示含有同樣的term，將其tf-idf相乘加起來
            if term_1 in dict2:
                tf_idf_1 = dict1[term_1] #dict1的value(tf-idf)
                tf_idf_2 = dict2[term_1] #dict2的value(tf-idf)
                count += float(tf_idf_1*tf_idf_2) #兩者相乘，加起來
        return count #回傳結果        

    def countCosineSimilarity(self,doc1,doc2):    #計算cosine similarity
        dict_1 = dict_tf_idf[int(doc1)] #key為term,value為tf-idf
        dict_2 = dict_tf_idf[int(doc2)] #key為term,value為tf-idf
        answer = self.cosineSimilarity(dict_tf_idf,dict_1,dict_2)
        return answer