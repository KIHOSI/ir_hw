import math

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

#讀取TermFrequency每一個doc的term的tf
for i in range(1,2):
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
    vectorLength = countDocVectorLength(dict_now)    
    for key,tf in dict_now.items(): #count tf-idf
        t_index,tf_idf = findIDF(key,tf)
        
        sqrt_tfidf = float(tf_idf/vectorLength)
        #print("sqrt_tfidf:%f" % sqrt_tfidf)

        with open('TF_IDF/'+str(i)+'.txt','a') as f: #t_index,tf-idf unit vector
            f.write("%d %f\n" % (t_index,sqrt_tfidf))
  