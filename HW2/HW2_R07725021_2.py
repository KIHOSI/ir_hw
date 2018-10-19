import math

def findIDF(dict,key,value): #去dictonary.txt找到該term的df和其t_index，將df變idf    
    #在dictionary.txt找到該term的df，將df變idf
    df = dict.get(key)
    idf = math.log10(1095/float(df))
    tf_idf = float(value)*float(idf)
    return float(tf_idf)

def countDocVectorLength(dict): #計算vector length
    squareNum = 0
    for key,value in dict.items():
        squareNum += int(value) * int(value) #tf^2相加
    sqrtNum = math.sqrt(squareNum) #最後squart roots
    return sqrtNum     


dict_df = {} #儲存dictionary的term和df
dict_index = {} #儲存dictionary的term和t_index
#取得dictionary.txt的值轉成dict
with open('dictionary.txt','r') as f:
    for line in f: 
        (t_index,key,value) = line.split()
        dict_df[key] = value #儲存term,df
        dict_index[key] = t_index #儲存term,t_index

#讀取TermFrequency每一個doc的term的tf
for i in range(1,1096):
    dict_now = {} # 儲存該doc的term和tf

    # doc i's file to dictionary
    with open('TermFrequency/'+str(i)+'.txt','r') as f:
        for line in f: #一行一行儲存
            (key,value) = line.split()
            dict_now[key] = value

    # 先清除i.doc檔案，寫上該doc含有多少term
    open('TF_IDF/'+str(i)+'.txt','w').close() # clear doc contents
    with open('TF_IDF/'+str(i)+'.txt','w') as f:
        f.write(str(len(dict_now))+"\n\n") # how many terms in this doc

    vectorLength = countDocVectorLength(dict_now) #計算vector length   
    for key,tf in dict_now.items(): #count tf-idf
        tf_idf = findIDF(dict_df,key,tf)
        t_index = int(dict_index.get(key))
        sqrt_tfidf = float(tf_idf/vectorLength) #get tf-idf unit vector

        with open('TF_IDF/'+str(i)+'.txt','a') as f: #t_index,tf-idf unit vector
            f.write("%d %f\n" % (t_index,sqrt_tfidf))
  