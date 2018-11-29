from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def tokenize(docid): #token
    file = open("IRTM/"+docid+".txt","r")
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

# def featureSelection():


dict_doc = {} #儲存tokenize後的1095文章的token
for i in range(1,2): #將1095個doc做token
    dict_doc[i] = tokenize(str(i)) #key為docid,value為該doc的所有token (docid轉成string)
    #print("dict_doc key:"+str(i)+"\n")
    #print(dict_doc)

#讀取training.txt，得到各class及對應的training docid
dict_class = {}
with open('training.txt','r') as f:
    for line in f:
        #print(line.split(' ',1))
        (classID,docID_list) = line.split(' ',1)
        docID_set = docID_list.split() #去除docid中最後面\n    
        dict_class[classID] = docID_set
# print(dict_class) 
#print("\n")
#token training document
dict_train_doc = {} #key為docid,value為tokens
for key,value in dict_class.items(): # key為classid,value為docid_list
    for docid in value:
        tokens = tokenize(docid)
        dict_train_doc[docid] = tokens

# print(dict_train_doc.keys())
#把dict_train_doc當成dict_class的value
dict_class_doc={}
for classid,docid_list in dict_class.items():
     for docid in docid_list:
        dict_test = {} #儲存dict_train_doc的docid和tokens
        dict_test[docid] = dict_train_doc[docid]
        dict_class_doc.setdefault(classid,[]) #預設是(key,[])，後面就可用append
        dict_class_doc[classid].append((dict_test)) #key為classid，value為docid和tokens
        
# print(dict_class_doc[str(1)])
#feature selection

