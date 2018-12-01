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

def countChiSquare(list_p,list_c,doc_all): #計算該term在該class的chi square總和（p-c,p-notc,a-c,a-notc)
    chi_square_sum = 0 #總和
    for i in range(0,2): #p:0,1
        for j in range(0,2): #c:0,1
            # print("list_p:"+str(i)+" "+str(list_p[i]))
            # print("list_c:"+str(j)+" "+str(list_c[j]))
            # print("doc_all:"+str(doc_all))
            expected_count = doc_all*(list_p[i]/doc_all)*(list_c[j]/doc_all)
            # print("expected_count:"+ str(expected_count))
            chi_square = ((doc_all - expected_count)**2)/expected_count
            # print("chi_square:"+ str(chi_square))
            chi_square_sum += chi_square
    # print("chi_square_sum:"+ str(chi_square_sum))
    return chi_square_sum

def chiSquare(term,dict_class,dict_train_doc): #計算每個term在13個class各別的chi-square，回傳13個裡面最大的值！
    #計算個chi squares需要的值（c-p,c-a,notc-p,notc-a,c-all,p-all,doc_all)
    doc_all = 195
    c_all = 15
    notc_all = 180

    #計算在每一個class各出現幾個doc(df) [class-present]
    c_p = []
    for classid,docid_list in dict_class.items(): #key為classid,value為docid和terms
        count_df = 0
        for docid in docid_list: #取每一個docid
            if term in dict_train_doc[docid]:#看terms list有沒有含這個term，有就代表這個class中這個doc有含這個term 
                count_df += 1
        c_p.append(count_df)  #這個term在這個class的df值

    p_all = 0 #[all present]
    c_a = [] #[class-absent]
    notc_p = [] #[not class-present]
    notc_a = [] #[not class-absent]
    a_all = 0 #[absent-all]

    for i in range(0,13): #第幾個class
        p_all += c_p[i] #得p_all
        c_a.append(15-c_p[i]) #c_all

        count_notc_p = 0 #計算各個class中not c-present的值
        for j in range(0,13): #加總not c-present總合
            if(j == i):
                continue;
            count_notc_p += c_p[j]
        notc_p.append(count_notc_p)
        notc_a.append(notc_all - notc_p[i])
    a_all += doc_all - p_all

    #這個算法為計算該term在全部class的4個chi square值，加總！（不確定對不對ＱＱ）
    list_p = [p_all,a_all]
    list_c = [c_all,notc_all]
    chi_square_sum = countChiSquare(list_p,list_c,doc_all)   


    return c_p  

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
print(chiSquare("navi",dict_class,dict_train_doc))
# print(dict_train_doc[str(11)])
# for term in dict_train_doc[str(11)]: #key為docid,value為terms，取得每一個term
#     chisquare = chiSquare(term,dict_class,dict_train_doc)
#     dict_chisquare[term] = chisquare
# print(dict_chisquare)
