import numpy as np
import matplotlib.pyplot as plt

#定義g(x)函數
def g(x):
    return np.where((x >= 0) & (x <= 2), 1 - np.abs(1-x), 0)

#生成符合機率分布的隨機變量
#使用向量化的方式來產生隨機數，並使用np.concatenate()來合併多個隨機樣本
def rejection_sampling(n):
    x = np.random.uniform(0, 10, size=n) #產生n個[0, 10]區間內的均勻分布隨機數
    y = np.random.uniform(0, 1, size=n) #產生n個[0, 1]區間內的均勻分布隨機數
    accept = y <= g(x) #判斷哪些x滿足接受條件
    samples = x[accept] #取出接受的隨機樣本
    while len(samples) < n: #如果接受的樣本數小於n
        x = np.random.uniform(0, 10, size=n-len(samples)) #再產生n-len(samples)個新的x隨機數
        y = np.random.uniform(0, 1, size=n-len(samples)) #再產生n-len(samples)個新的y隨機數
        accept = y <= g(x) #判斷哪些新的x滿足接受條件
        new_samples = x[accept] #取出新的接受樣本
        samples = np.concatenate((samples, new_samples)) #將新的接受樣本合併到原來的樣本中
    return samples[:n] #返回n個隨機樣本

#每次隨機抽樣 n 次 for n = 10, 50, 100, 200
n = [10, 50, 100, 200]
variances = [] 
for i in n:
    samples = rejection_sampling(50000 * i)#使用 rejection_sampling 函數生成 50000 個符合機率分布的隨機變量，並將這些變量儲存到 samples 中
    samples = samples.reshape((50000, i))#將 samples 重塑為一個形狀為 (50000, i) 的矩陣，其中每行都包含 i 個隨機變量
    sample_var = np.var(samples, axis=1)#計算每行的變異數，並將這些變異數儲存到 sample_var 中
    variances.append(sample_var) # 把50000次樣本變異數的平均值加入到變異數列表中
    
# 繪製次數分配圖
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))  # 創建一個 2x2 的子圖 
axs = axs.flatten()
for i in range (len(n)):
    axs[i].hist(variances[i], bins = 50, alpha = 0.5) #繪製長條圖
    axs[i].set_title(f"n = {n[i]}")     
plt.show()