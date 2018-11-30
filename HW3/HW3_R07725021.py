from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def tokenize(docid): #term
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

    return removeStopword #回傳tokenize後該文章的term（字串）

def chiSquare(term,dict_class,dict_train_doc): #計算每個term在13個class各別的chi-square，回傳13個裡面最大的值！
    #先計算在每一個class各出現幾個doc(df)
    c_present = []
    for classid,docid_list in dict_class.items(): #key為classid,value為docid和terms
        count_df = 0
        for docid in docid_list: #取每一個docid
            if term in dict_train_doc[docid]:#看terms list有沒有含這個term，有就代表這個class中這個doc有含這個term 
                count_df += 1
        c_present.insert(int(classid),count_df)  #這個term在這個class的df值

    # print("c_present:")
    # print(c_present)
    return c_present     

dict_doc = {} #儲存tokenize後的1095文章的term
for i in range(1,2): #將1095個doc做tokenize
    dict_doc[i] = tokenize(str(i)) #key為docid,value為該doc的所有term (docid轉成string)
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
#tokenize training document
dict_train_doc = {} #key為docid,value為terms
for key,value in dict_class.items(): # key為classid,value為docid_list
    for docid in value:
        tokens = tokenize(docid)
        dict_train_doc[docid] = tokens

#feature selection
#計算每個train doc的term的chi-square
dict_chisquare = {} #key為term,value為chisquare值
# chiSquare("navi",dict_class,dict_train_doc)
# print("\nchisquare:"+chisquare)
print(dict_train_doc[str(11)])
for term in dict_train_doc[str(11)]: #key為docid,value為terms，取得每一個term
    chisquare = chiSquare(term,dict_class,dict_train_doc)
    dict_chisquare[term] = chisquare
print(dict_chisquare)
