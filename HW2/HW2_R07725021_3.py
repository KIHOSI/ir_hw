def readFileToDict(num): #選擇要load哪個doc，並將內容換成dict
    with open('TF_IDF/'+str(num)+'.txt','r') as f:
        for line in f:
            if not((key,value) = line.split()):
                pass
            else:
                (key,value) = line.split()
                dict[key] = value  
        return dict              

def cosineSimilarity(dict1,dict2):
    


doc1 = input("要比較哪兩個文章的cosine similarity(1):")
doc2 = input("要比較哪兩個文章的cosine similarity(2):")
dict_1 = readFileToDict(doc1)
dict_2 = readFileToDict(doc2)
cosineSimilarity(doc1,doc2)