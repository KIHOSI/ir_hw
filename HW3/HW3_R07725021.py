from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def tokenize(docid): #token
    file = open("IRTM/"+str(docid)+".txt","r")
    data = file.read()
    file.close()

    #tokenize(用nltk的tokenizer)
    tokendata = word_tokenize(data)
    regex = [c for c in tokendata if c.isalpha()] #去除non-alphanumeric

    #lowercasing
    lowerString = [elements.lower() for elements in regex]

    #stemming
    stemmer = PorterStemmer()
    stemString = [stemmer.stem(lowerstring) for lowerstring in lowerString]
    
    #stopword removal
    file2 = open("stopwordlist.txt","r")
    stopwordList = file2.read()
    file2.close()
    removeStopword = [word for word in stemString if word not in stopwordList]

    return removeStopword #回傳tokenize後該文章的token（字串）

dict_doc = {} #儲存tokenize後的1095文章的token
for i in range(1,2): #將1095個doc做token
    dict_doc[i] = tokenize(i) #key為docid,value為該doc的所有token
    #print("dict_doc key:"+str(i)+"\n")
    #print(dict_doc)

#讀取training.txt，得到各class及對應的training docid
with open('training.txt','r') as f:
    for line in f:
        


