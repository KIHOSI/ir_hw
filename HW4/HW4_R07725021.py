from CountSimilarity import CountSimilarity #import CountSimilarity.py
from collections import defaultdict
import numpy as np

def simpleHAC():
    clusterSimilarity = defaultdict(dict) #儲存任兩個文章的cosine similarity
    existCluster = np.ones(1096) #每個cluster是否還存活著，1存在、0不存在，預設都存在
    clusterArray = [] #存現在merge的文章們(list)

    #得N*N matrix，計算任兩文章的similarity
    for i in range(1,1096):
        for j in range(1,1096):
            clusterSimilarity[i][j] = cosineSimilarity.countCosineSimilarity(i,j)

    #merge
    # for i in range(1,1095):
    for i in range(1,1096):
        print("i:"+str(i)+"\n")
        print(sorted(clusterSimilarity[i].items(),key = lambda x:x[1],reverse=True))
        for j,similarity in sorted(clusterSimilarity[i].items(),key = lambda x:x[1],reverse=True):
        # for j in range(1,1096):
            if((i != j) and (existCluster[i] == 1) and (existCluster[j] == 1)): #i不等於j，且i和j文章都還活著
                #取得跟i比，similarity最大的j
                # test = sorted(clusterSimilarity[i].items(),key = lambda x:x[1],reverse=True) #排序，比較similarity(value)，由大到小，取最大的
                print("j:"+str(j)+" ,similarity:"+str(similarity)+"\n")
                break
        break
                #看i比較大還是j比較大
                # if(i > j): #i比較大，與j對調
                #     num = i
                #     i = j
                #     j = num
                # clusterArray.append([i,j])
                #更新cosine similarity

# def completeLinkage(clusters,):




global cosineSimilarity
cosineSimilarity = CountSimilarity() #引用CountSimilarity.py class
cosineSimilarity.main() #初始化，先算出tf-idf值
dict_tf_idf = cosineSimilarity.getTFIDFDict()
# simpleHAC()
# print(clusterSimilarity[1][2])
# cosineSimilarity.countSimilarity(1,2)
# cosineSimilarity.countSimilarity(3,2)
# cosineSimilarity.countSimilarity(12,2)
# cosineSimilarity.countSimilarity(220,10)

