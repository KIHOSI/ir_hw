from CountSimilarity import CountSimilarity #import CountSimilarity.py
from collections import defaultdict
import numpy as np
# import pandas as pd

def simpleHAC():
    clusterSimilarity = np.zeros(shape=(1096,1096)) #浪費空間，index比較好看
    existCluster = np.ones(1096) #每個cluster是否還存活著，1存在、0不存在，預設都存在
    clusterDict = defaultdict(list) #存現在merge的文章們(dict型態，key為cluster的頭，value是list)
    for num in range(1,1096): #initial,key和value都是docid
        clusterDict[num].append(num)
    

    #得N*N matrix，計算任兩文章的similarity
    for i in range(1,1096):
        for j in range(1,1096):
            if(i == j): #避免self-similarity
                continue 
            clusterSimilarity[i][j] = cosineSimilarity.countCosineSimilarity(i,j)

    #merge，把每個doc都先看成一個點，任兩點的cosine similarity就是他們之間的距離，先跟據哪兩個點距離最小(sim最大)來合併變一群
    #距離 = 1-cosineSimilarity，所以cosine similarity大，距離短
    #合併過後，single linkage是看[i][j]和[m][j]哪個比較短(sim比較大)
    #complete linkage則是看[i][j]和[m][j]哪個比較長(sim比較小)
    for k in range(1,1095): # merge次數，每次都是兩兩merge
        if(len(clusterDict.keys())<=targetCluster): #決定要幾群
            break
        #取得i,m,similarity，比較similarity，拿sim最大的i和m來做merge
        max_list = np.dstack(np.unravel_index(np.argsort(-clusterSimilarity.ravel()), (1096, 1096)))#回傳array裡sim由大到小的index值，e.g. [563,594],[594,595],...
        for num in max_list[0]:#取得i,m    
            i = num[0]
            m = num[1]
            if((i != m) and (existCluster[i] == 1) and (existCluster[m] == 1)): #i不等於j，且i和j文章都還活著
            #取得跟i比，similarity最大的m
            # print("m:"+str(m)+" ,similarity:"+str(similarity)+"\n")
                    
                if(i > m): #i永遠是比較小的
                    num = i
                    i = m
                    m = num
                        
                #把j merge到i去
#                 if(clusterDict.get(m)):#如果m也是已有list，取得
                # clusterDict[i].append(m) #m本身也要加
                clusterDict[i].extend(clusterDict[m]) #m的底下list
#                 else: #m只是單個值
#                 clusterDict[i].append(m)

                clusterDict.pop(m, None) #如果dict裡有key為m，要刪掉(因為m已經merge到i去了)        
                #找sim最大的（距離最遠，complete linkage)
                for j in range(1,1096):
                    if((j!=i) and (j!=m)):
                        clusterSimilarity[i][j] = min(clusterSimilarity[i][j],clusterSimilarity[m][j])
                        clusterSimilarity[j][i] = min(clusterSimilarity[i][j],clusterSimilarity[m][j])
                existCluster[m] = 0
                break

    return clusterDict


global cosineSimilarity
global targetCluster #目標分成幾群
targetCluster = 20
cosineSimilarity = CountSimilarity() #引用CountSimilarity.py class
cosineSimilarity.main() #初始化，先算出tf-idf值
answer = simpleHAC()
print("dict keys num:"+str(len(answer.keys()))+"\n")
print(answer)
with open(str(targetCluster)+'txt','w') as f:
    for cluster_boss,cluster_subs in answer.items():
        # f.write("%d\n" % (cluster_boss))
        for cluster_sub in sorted(cluster_subs):
            f.write("%d\n" % (cluster_sub))
        f.write("\n\n")