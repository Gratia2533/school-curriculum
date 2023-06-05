import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import time

# 初始時間
a = time.time()
# 定義Bootstrap
def bootstrap(data, n_bootstrap):
    n = len(data)
    bootstrap_samples = np.zeros((n_bootstrap, n))
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_samples[i] = bootstrap_sample
    return bootstrap_samples

# 初始化MPI環境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 假設有一組原始數據
strong_hand = np.array([77.3, 95.1, 97.2, 94.5, 83.6, 90.2, 84.3])
weak_hand = np.array([74.6, 83.4, 80.6, 90.2, 78.7, 83.6, 76.2])

#計算原始的平均數差異
u_diff = np.mean(strong_hand) - np.mean(weak_hand)

# 將10個bootstrap樣本平均分配給每個CPU
n_bootstrap = 1000000
n_bootstrap_per_cpu = n_bootstrap // size

# 每個CPU執行自己分配到的bootstrap樣本
bootstrap_samples_local1 = bootstrap(strong_hand, n_bootstrap_per_cpu)
bootstrap_samples_local2 = bootstrap(weak_hand, n_bootstrap_per_cpu)

# 將每個CPU的結果收集到根節點
bootstrap_samples_all1= comm.gather(bootstrap_samples_local1, root=0)
bootstrap_samples_all2 = comm.gather(bootstrap_samples_local2, root=0)

if rank == 0:
    '''
    for i, samples in enumerate(bootstrap_samples_all):
        for j, sample in enumerate(samples):
            print(f"Bootstrap樣本{size*i + j + 1}：", sample)
    '''
    # 計算每個bootstrap樣本的平均數
    bootstrap_means1 = np.mean(bootstrap_samples_all1, axis=(0, 2))
    bootstrap_means2 = np.mean(bootstrap_samples_all2, axis=(0, 2))
    mean_diff = bootstrap_means1 - bootstrap_means2
    
    #計算std
    std = np.std(mean_diff) / np.sqrt(n_bootstrap)

    #計算信賴區間的上下界
    lower = u_diff - (1.96 * std)
    upper = u_diff + (1.96 * std)
    
    #利用信賴區間判斷平均數是否有顯著差異
    if 0 < lower or 0 > upper:
        decision = "reject the null hypothesis"
    else:
        decision = "don't reject the null hypothesis"

    # 輸出每一筆bootstrap樣本的平均數
    '''
    for i, mean in enumerate(bootstrap_means):
        print(f"Bootstrap樣本{size*i + rank + 1}的平均數：", mean)
    '''
    '''
    
    # 繪製值方圖
    plt.hist(bootstrap_means1, bins=30, edgecolor='black')
    plt.xlabel('Bootstrap Mean')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bootstrap_means1')
    plt.show()
    
    plt.hist(bootstrap_means2, bins=30 , edgecolor='black',)
    plt.xlabel('Bootstrap Mean')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bootstrap_means2')
    plt.show()
    
    plt.hist(mean_diff, bins=30 , edgecolor='black',)
    plt.xlabel('Bootstrap Mean')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bootstrap mean_diff')
    plt.show()
    '''
    print(f"difference of observed mean = {u_diff:.3f}")
    print(f"95% confidence interval：[{lower:.3f},{upper:.3f}]")
    print(f"decision: {decision}")
# 結束時間
b = time.time()
print(f"time {b-a:.3f}")
print(b-a)