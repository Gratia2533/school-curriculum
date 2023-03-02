import numpy as np
import pandas as pd
#%%計算兩點間的歐幾里德距離
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(4):
        distance += (point1[i] - point2[i]) ** 2
    return np.sqrt(distance)
#%%
def k_means(data, K):
    #隨機選擇初始點
    np.random.seed(10)
    centroids = data[np.random.choice(data.shape[0], K, replace=False), :]

    #初始化空list，去存cluster和距離
    clusters = []
    distances = []

    #重複執行直到收斂
    while True:
        #初始化空list，去存每個點的距離和其分配到的cluster
        point_distances = []
        point_clusters = []

        #迭代每個點
        for point in data:
            #計算點到每個中心點的距離
            distance = [euclidean_distance(point, centroid) for centroid in centroids]
            #把點分配給距離中心點最近的那個群
            cluster = np.argmin(distance)
            #存該點的距離和它的群
            point_distances.append(distance[cluster])
            point_clusters.append(cluster)

        #更新cluster和距離
        clusters = point_clusters
        distances = point_distances

        #初始化空list去存新的中心點
        new_centroids = []

        #在所有群上迭代
        for i in range(K):
            #取得當前cluster的所有資料點
            points = data[np.where(np.array(clusters) == i)[0], :]
            #計算點的平均值得到中心點
            centroid = np.mean(points, axis=0)
            #增加中心點到list
            new_centroids.append(centroid)

        #確認中心點是否有變化
        if np.array_equal(centroids, new_centroids):
            #如果沒有變化代表收斂穩定
            break
        else:
            #如果有改變，則繼續更新中心點去迭代
            centroids = new_centroids

    #回傳clusters跟中心點
    return clusters, centroids
#%%計算輪廓係數
def silhouette_coefficient(X, labels):
    n_samples, _ = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    a = np.zeros(n_samples)
    b = np.zeros(n_samples)
    
    for i in range(n_samples):
        cluster_i = labels[i]
        
        # 計算同一群中與其他所有點的平均距離
        a[i] = np.mean([euclidean_distance(X[i], X[j]) for j in range(n_samples) if labels[j] == cluster_i])
        
        # 將b計算為到最近的其他cluster中所有點的平均距離
        min_b = np.inf
        for k in unique_labels:
            if k != cluster_i:
                cluster_k = np.mean([euclidean_distance(X[i], X[j]) for j in range(n_samples) if labels[j] == k])
                if cluster_k < min_b:
                    min_b = cluster_k
        b[i] = min_b
    
    # 計算所有點的輪廓係數
    s = (b - a) / np.maximum(a, b)
    
    # 回傳平均值
    return np.mean(s)
#%%匯入資料
initial_data = pd.read_csv('MLHW1.csv')
data = []#建立空陣列儲存點
for i in range(0,75):
    data.append(list(initial_data.loc[i]))
print('共有',len(data),'個點')
print(data)
data = np.array(data)
#%%
K = 4
clusters, centroids = k_means(data, K)
silhouette = silhouette_coefficient(data , clusters)
print('輪廓係數為:',silhouette)
print('中心點:\n',centroids)
print('各點分布的群:\n',clusters)
from collections import Counter
counts = Counter(clusters)
print('各群含有的點的數量:',counts)

#%%觀察分為幾群時的輪廓係數，在K=4有最佳表現
s = []
k_cluster = []
for K in range(2,11):
    clusters, centroids = k_means(data, K)
    silhouette = silhouette_coefficient(data , clusters)
    s.append(silhouette)
    k_cluster.append(K)
    print('當分為',K,'群,輪廓係數為:',silhouette)
#%%建立輪廓係數的圖表
chart = pd.DataFrame({'Silhouette':s},index = k_cluster)
import matplotlib.pyplot as plt
plt.plot(chart.index, chart.Silhouette,'b',label='Silhouette')
plt.xlabel('K')
plt.ylabel('Silhouette')
plt.xlim(2,10)
plt.ylim(0.35,0.7)
plt.show