from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# import numpy as np
from collections import defaultdict
import math

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

    #這個算法為計算該term在全部class的4個chi square值，加總！（就算個別算，值也一樣）
    #因為四個值會加總，最終p_all,c_all,a_all,notc_all都一樣
    list_p = [p_all,a_all]
    list_c = [c_all,notc_all]
    chi_square_sum = countChiSquare(list_p,list_c,doc_all)   
    # print(chi_square_sum)

    return chi_square_sum 

def trainMultinomialNB(dict_class,dict_train_doc_filter,feature_selection_list): #Multinomail model for training phase
    # test[1][1] = 1

    Nc = 15 # 每個class有15個train doc
    N = 195 # 13個class，有13*15 = 195個train doc
    prior_c = [] #儲存每個class的P(c)
    prior_c.insert(0,0) #index從1開始
    condprob =  defaultdict(dict) #two dimensional dict

    for classid,docid_list in dict_class.items():
        prior_c.insert(int(classid),(Nc/N)) #P(c) = Nc / N
        text_c = 0 #該class所有terms數(textc)
        text_term = 0 #每個term在該class的doc中總共出現幾次(Tct)
        for docid in docid_list: #取得每個class的docid list，得每一個doc
            text_c += len(dict_train_doc_filter[docid]) #計算這個class總terms數
            #計算每個term in V(feature selection的500個terms)在這個class中所有doc裡出現幾次
            for term in feature_selection_list:
                dict_train_doc_filter[docid].count(term)
        
        #建condprob[t][c]
        # condprob = [][]
        # condprob = np.zeros((len(feature_selection_list),len(dict_class.keys())))
        
        for term in feature_selection_list:
            condprob[term][classid] = (text_term+1)/(text_c+len(feature_selection_list)) # add one smoothing，M為feature selection的總terms數(500)
            # print(condprob['distress'])
            # condprob.setdefault(term,{})[classid] = (text_term+1)/(text_c+len(feature_selection_list)) # add one smoothing，M為feature selection的總terms數(500)  

    # print(feature_selection_list)  
    # print(condprob)
    # print(prior_c)  
    # print(condprob['distress'])

    return condprob,prior_c

def ApplyMultinomialNB(dict_class,test_data,condprob,prior_c): # multinomial model in testing phase
    #判斷該test doc屬於哪一個class
    score = [] #儲存term在每一個class的score,score = logP(c) + logP(X = t | c)
    score.insert(0,0) #index從1開始
    for classid,docid_list in dict_class.items():
        score.insert(int(classid),math.log(prior_c[int(classid)]))
        for term in test_data: #該test doc的term，在這個class的分數加總，就是這個doc屬於這個class的分數
            # print("term:"+term+"\n")
            score[int(classid)] += math.log(condprob[term][classid])
    print(score) 




dict_doc = {} #儲存tokenize後的1095文章的term
for i in range(1,1096): #將1095個doc做tokenize
    dict_doc[str(i)] = tokenize(str(i)) #key為docid,value為該doc的所有term (i和docid轉成string)
    #print("dict_doc key:"+str(i)+"\n")
    #print(dict_doc)

#讀取training.txt，得到各class及對應的training docid
dict_class = {} # key為classid,value為docid_list
with open('training.txt','r') as f:
    for line in f:
        #print(line.split(' ',1))
        (classID,docID_list) = line.split(' ',1)
        docID_set = docID_list.split() #去除docid中最後面\n    
        dict_class[classID] = docID_set

#tokenize training document
dict_train_doc = {} #key為docid,value為terms
for key,value in dict_class.items(): # key為classid,value為docid_list
    for docid in value:
        tokens = tokenize(docid)
        dict_train_doc[docid] = tokens

#去除train doc，剩下1095-195=900個test doc
# print(len(dict_doc))
dict_test_doc = dict_doc.copy() #複製1095個doc
# print(len(dict_test_doc))
for classid,docid_list in dict_class.items():
    for docid in docid_list:
        del dict_test_doc[docid] #刪除在train data的doc，留下來都當test data
        # for docid,terms in dict_doc.items():
        #     if not(docid == docid2):
        #         dict_test_doc[docid] = terms
# print(len(dict_test_doc.keys()))

#feature selection
#計算每個train doc的term的chi-square
dict_chisquare = {} #key為term,value為chisquare值
# print(chiSquare("navi",dict_class,dict_train_doc))
# print(dict_train_doc[str(11)])
for terms_list in dict_train_doc.values(): #key為docid,value為terms，取得每一個term
    for term in terms_list:
        # print(term)
        chisquare = chiSquare(term,dict_class,dict_train_doc)
        if(dict_chisquare.get(term)): #有重複的term,跳過
            continue
        dict_chisquare[term] = chisquare

#取前500個chi-square大的term
feature_selection_list = []
count = 1
for key,value in sorted(dict_chisquare.items(), key = lambda x:x[1],reverse=True): #sorted by value(由大到小)
    # print("%s %s\n" % (key,value))
    if(count > 500):
        break
    feature_selection_list.append(key)
    count += 1    

# print(len(feature_selection_list))
# print(feature_selection_list)

#過濾train data和test data，只剩這500個term
dict_train_doc_filter = {} #key為docid,value為terms
dict_test_doc_filter = {} #key為docid,value為terms
for docid,terms in dict_train_doc.items(): #將train data過濾 
    filter_terms = [term for term in terms if term in feature_selection_list]  #只留下在feature selection後這500個term的字
    dict_train_doc_filter[docid] = filter_terms

for docid,terms in dict_test_doc.items(): #將test data過濾
    filter_terms = [term for term in terms if term in feature_selection_list]  #只留下在feature selection後這500個term的字
    dict_test_doc_filter[docid] = filter_terms

# print(dict_test_doc_filter)

#分類
condprob,prior_c = trainMultinomialNB(dict_class,dict_train_doc_filter,feature_selection_list) #train data
#test data，算出每一個doc屬於哪一個class
dict_answer = {}
ApplyMultinomialNB(dict_class,["distress","lesli"],condprob,prior_c)
# for docid,terms in sorted(dict_test_doc_filter.items()):
#     doc_class = ApplyMultinomialNB(dict_class,terms,condprob,prior_c) #得到該doc屬於哪個class
#     dict_answer[int(docid)] = int(doc_class)


