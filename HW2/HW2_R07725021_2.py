import math

def findIDF(key,value): #去dictonary.txt找到該term的df和其t_index，將df變idf
    with open('dictionary.txt','r') as f:
        exit_flag = False #用以跳出for迴圈
        for line in f:
            (index_df,key_df,df) = line.split()
            if(key == key_df):
                idf = math.log10(1095/float(df))    
                tf_idf = float(value)*float(idf)
                exit_flag = True
                return int(index_df),float(tf_idf)
            if(exit_flag):
                break          

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

    for key,tf in dict_now.items(): #count tf-idf
        t_index,tf_idf = findIDF(key,tf)

        with open('TF_IDF/'+str(i)+'.txt','a') as f: #t_index,tf-idf
            f.write("%d %f\n" % (t_index,tf_idf))
  