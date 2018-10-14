import math

def findIDF(key,value): #去dictonary.txt找到該term的df和其t_index，將df變idf
    with open('dictionary.txt','r') as f:
        for line in f:
            (index_df,key_df,df) = line.split()
            #print("%s %s %s\n" % (index_df,key_df,df))
            if(key == key_df):
                #print("df:%f\n" % float(df))
                #print("1095/df: %f" % (1095/float(df)))
                #print("log10(1095/df): %f" % (math.log(1095/float(df))))
                idf = math.log10(1095/float(df))
                #print("idf:%f\n" % idf)
                tf_idf = float(value)*float(idf)
                #print("tf-idf:%f\n" % tf_idf)
                return int(index_df),float(tf_idf)      

#讀取TermFrequency每一個doc的term的tf
for i in range(1,2):
    dict_now = {} # 儲存該doc的term和tf
    # file to dictionary
    with open('TermFrequency/'+str(i)+'.txt','r') as f:
        for line in f: #一行一行儲存
            (key,value) = line.split()
            dict_now[key] = value

    #去dictonary.txt找到該term的df和其t_index，將df變idf
    #print("dict_now: %d" % len(dict_now))
    #print("dict_now(key): %d" % len(dict_now.keys()))
    open('TF_IDF/'+str(i)+'.txt','w').close() # clear doc contents
    with open('TF_IDF/'+str(i)+'.txt','w') as f:
        f.write(str(len(dict_now))+"\n\n") # how many terms in this doc

    for key,tf in dict_now.items():
        t_index,tf_idf = findIDF(key,tf)

        with open('TF_IDF/'+str(i)+'.txt','a') as f: #t_index,tf-idf
            f.write("%d %f\n" % (t_index,tf_idf))
  