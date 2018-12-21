from CountSimilarity import CountSimilarity #import CountSimilarity.py
from collections import defaultdict
import numpy as np

def simpleHAC():
    clusterSimilarity = defaultdict(dict) #儲存任兩個文章的cosine similarity
    existCluster = np.ones(1096) #每個cluster是否還存活著，1存在、0不存在，預設都存在
    clusterArray = {} #存現在merge的文章們(dict)

    #得N*N matrix，計算任兩文章的similarity
    for i in range(1,1096):
        for j in range(1,1096):
            clusterSimilarity[i][j] = cosineSimilarity.countCosineSimilarity(i,j)

    #merge，把每個doc都先看成一個點，任兩點的cosine similarity就是他們之間的距離，先跟據哪兩個點距離最小(sim最大)來合併變一群
    #距離 = 1-cosineSimilarity，所以cosine similarity大，距離短
    #合併過後，single linkage是看[i][j]和[m][j]哪個比較短(sim比較大)
    #complete linkage則是看[i][j]和[m][j]哪個比較長(sim比較小)
    for k in range(1,8): # K = 8（剩8群，?)
        for i in range(1,1096):
            # print("i1:"+str(i)+"\n\n")
            # print(sorted(clusterSimilarity[i].items(),key = lambda x:x[1],reverse=True))
            # m = 0 #initial
            # similarity = 0 #initial
            for m,similarity in sorted(clusterSimilarity[i].items(),key = lambda x:x[1],reverse=True): #排序，比較similarity(value)，由大到小，取最大的
            # print("i2:"+str(i)+"\n\n")
                if((i != m) and (existCluster[i] == 1) and (existCluster[m] == 1)): #i不等於j，且i和j文章都還活著
                    # print("i3:"+str(i)+"\n\n")
                    #取得跟i比，similarity最大的m
                    # print("m:"+str(m)+" ,similarity:"+str(similarity)+"\n")
                    #把j merge到i去
                    if(i > m): #i永遠是比較小的
                        num = i
                        i = m
                        m = num
                    
                    cluster_list = m
                    if(clusterArray.get(i)):
                        clusterArray[i] += m
                    else:
                        clusterArray[i] = m

                    #找sim最大的（距離最遠，complete linkage)
                    for j in range(1,1096):
                        # print("i5:"+str(i)+"\n")
                        # print("j:"+str(j)+"\n")
                        # print("m:"+str(m)+"\n")
                        if((j!=i) and (j!=m)):
                            clusterSimilarity[i][j] = min(clusterSimilarity[i][j],clusterSimilarity[m][j])
                            clusterSimilarity[j][i] = min(clusterSimilarity[i][j],clusterSimilarity[m][j])
                    existCluster[m] = 0
                    break

    return clusterArray

                # break
        # break
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
# dict_tf_idf = cosineSimilarity.getTFIDFDict()
answer = simpleHAC()
with open('answer.txt','w') as f:
    for cluster_boss,cluster_subs in answer.items():
        f.write("%d\n" % (cluster_boss))
        for cluster_sub in cluster_subs:
            f.write("%d\n" % (cluster_sub))
    f.write("\n\n")

