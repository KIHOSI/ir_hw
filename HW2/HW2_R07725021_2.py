import math
import numpy as np
import pandas as pd

def findIDF(key,value): #去dictonary.txt找到該term的df和其t_index，將df變idf
    with open('dictionary.txt','r') as f:
        exit_flag = False #用以跳出for迴圈
        for line in f:
            (index_df,key_df,df) = line.split()
            if(key == key_df):
                #print("df:%f\n" % float(df))
                #print("1095/df: %f" % (1095/float(df)))
                #print("log10(1095/df): %f" % (math.log10(1095/float(df))))
                idf = math.log10(1095/float(df))    
                tf_idf = float(value)*float(idf)
                #print("tf-idf:%f\n" % tf_idf)
                exit_flag = True
                return int(index_df),float(tf_idf)
            if(exit_flag):
                break          

def countDocVectorLength(dict): #計算vector length
    squareNum = 0
    for key,value in dict.items():
        squareNum += int(value) * int(value) #tf^2相加
    sqrtNum = math.sqrt(squareNum) #最後squart roots
    #print("vector length:%f\n" % sqrtNum)
    return sqrtNum     

# def countTF(term): #用dictionary去找1~1095.txt中各term的tf，用成向量圖
#     tfArray = [] #儲存該term在所有1095個doc中的tf值
#     for i in range(1,1096): #要跑1095次QQ
#         with open('TermFrequency/'+str(i)+'.txt','r') as f:
#             exit_flag = False #用以跳出for迴圈
#             for line in f:
#                 (t,tf) = line.split()
#                 if(term == t): #找到term，回傳該doc和term的tf值
#                     tfArray.append(tf)
#                     exit_flag = True
#                 elif(term < t): #有排序過，所以如果已經找到比較大的，代表沒了
#                     tfArray.append(0)
#                     exit_flag = True
#                 if(exit_flag):
#                     break      
#     return tfArray


# with open('dictionary.txt','r') as f:
#     tfArray = []
#     termArray = []
#     tfAllArray = []
#     x = []
#     for line in f: # 2萬多次
#         (t_index,term,df) = line.split()
#         termArray.append(term)
#         tfArray = countTF(term)
#         x = np.append(x,tfArray)
    
    
#     index = pd.DataFrame(x,index=termArray,columns=range(1,1096))
#     print(index.shape)    


#讀取TermFrequency每一個doc的term的tf
for i in range(1,1096):
    dict_now = {} # 儲存該doc的term和tf
    # file to dictionary
    with open('TermFrequency/'+str(i)+'.txt','r') as f:
        for line in f: #一行一行儲存
            (key,value) = line.split()
            dict_now[key] = value

    #去dictonary.txt找到該term的df和其t_index，將df變idf
    open('TF_IDF/'+str(i)+'.txt','w').close() # clear doc contents
    with open('TF_IDF/'+str(i)+'.txt','w') as f:
        f.write(str(len(dict_now))+"\n\n") # how many terms in this doc

    #print("doc %d" %i)    
    vectorLength = countDocVectorLength(dict_now) #計算vector length   
    for key,tf in dict_now.items(): #count tf-idf
        t_index,tf_idf = findIDF(key,tf)
        
        sqrt_tfidf = float(tf_idf/vectorLength) #get tf-idf unit vector
        #print("sqrt_tfidf:%f" % sqrt_tfidf)

        with open('TF_IDF/'+str(i)+'.txt','a') as f: #t_index,tf-idf unit vector
            f.write("%d %f\n" % (t_index,sqrt_tfidf))
  