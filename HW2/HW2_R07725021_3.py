def readFileToDict(num): #選擇要load哪個doc，並將內容換成dict
    with open('TF_IDF/'+str(num)+'.txt','r') as f:
        dict = {}
        for index,line in enumerate(f,start=1): #(x,y,z...)變成有索引值的list(1,x),(2,y)
            if(index>=3):
                #print("line:%s" %line)
                (key,value) = line.split()
                dict[int(key)] = float(value)
        return dict                 

def cosineSimilarity(dict1,dict2):
    count = 0
    for key_1 in dict1: #比較dict2是否有包含dict1的key，有表示含有同樣的term，將其tf-idf相乘加起來
        if key_1 in dict2:
            value_1 = dict1[key_1] #dict1的value(tf-idf)
            value_2 = dict2[key_1] #dict2的value(tf-idf)
            count += float(value_1*value_2) #兩者相乘，加起來
    return count #回傳結果        



doc1 = input("要比較哪兩個文章的cosine similarity(1):")
doc2 = input("要比較哪兩個文章的cosine similarity(2):")
dict_1 = readFileToDict(doc1)
dict_2 = readFileToDict(doc2)
answer = cosineSimilarity(dict_1,dict_2)
print("doc"+str(doc1)+"與doc"+str(doc2)+"的cosine similarity為: "+str(answer)+" !!!")