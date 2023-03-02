import pandas as pd
from mlxtend.frequent_patterns import apriori
#%%讀取資料
vocab = pd.read_fwf("vocab.kos.txt",header=None)
vocab.index = vocab.index+1
vocab = vocab[0]#轉成series格式
docword = pd.read_csv("docword.kos.txt",sep=" ",header=None)
docword.columns=list('DWN')#修改對應欄位名稱
#%%把文檔依words分類
for i in range(1,3431):
    globals()['doc_'+str(i)] = docword[docword['D']==i].transpose()
    globals()['doc_'+str(i)].columns = list(globals()['doc_'+str(i)].loc['W'])
    globals()['doc_'+str(i)] = globals()['doc_'+str(i)].drop(index = 'D')
    globals()['doc_'+str(i)] = globals()['doc_'+str(i)].drop(index = 'W')
    globals()['doc_'+str(i)].index = pd.Series(['doc_'+str(i)])
#%%重新整併資料
newData = pd.concat([doc_1,doc_2], join='outer')
for j in range(3,3431):    
    newData = pd.concat([newData,globals()['doc_'+str(j)]], join = 'outer')
newData.fillna(value=0, inplace=True)
newData = newData.astype(int)
#%%將資料以向量方式輸出
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
vectorData = newData.applymap(encode_units)
vectorData.columns = list(vocab)#將欄位以對應單詞顯示
#%%建立空list去存後面要記錄的執行時間
Alltime=[]
Cltime=[]
Maxtime=[]
#%%利用Apriori演算法
import time
start_time=time.time()
All = apriori(vectorData,min_support=0.3,use_colnames=True)
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))
All_time = time.time()-start_time
print('All frquent itemsets\n',All)
#%%將該輪的執行時間存入對應的list
Alltime+=[All_time]
#%%Closed&Maximal
su = All.support.unique()#all unique support count
#Dictionay storing itemset with same support count key
fredic = {}
for i in range(len(su)):
    inset = list(All.loc[All.support ==su[i]]['itemsets'])
    fredic[su[i]] = inset
#Dictionay storing itemset with  support count <= key
fredic2 = {}
for i in range(len(su)):
    inset2 = list(All.loc[All.support<=su[i]]['itemsets'])
    fredic2[su[i]] = inset2

#%%Find Closed frequent itemset
start_time = time.time()
cl = []
for index, row in All.iterrows():
    isclose = True
    cli = row['itemsets']
    cls = row['support']
    checkset = fredic[cls]
    for i in checkset:
        if (cli!=i):
            if(frozenset.issubset(cli,i)):
                isclose = False
                break
    
    if(isclose):
        cl.append(row['itemsets'])
print('Time to find Close frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))  
Closed_time = time.time()-start_time
print('Closed frquent itemsets\n',cl)
#%%將該輪的執行時間存入對應的list
Cltime+=[Closed_time]
#%%Find Max frequent itemset
start_time = time.time()
ml = []
for index, row in All.iterrows():
    isclose = True
    cli = row['itemsets']
    cls = row['support']
    checkset = fredic2[cls]
    for i in checkset:
        if (cli!=i):
            if(frozenset.issubset(cli,i)):
                isclose = False
                break
    
    if(isclose):
        ml.append(row['itemsets'])
print('Time to find Max frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))
Maximal_time = time.time()-start_time
print('Maximal frquent itemsets\n',ml)
#%%將該輪的執行時間存入對應的list
Maxtime+=[Maximal_time]
#%%建立執行時間
index_name = [0.15,0.2,0.25,0.3,0.35]
timeData = pd.DataFrame({'All':Alltime,'Closed':Cltime,'Maximal':Maxtime},index = index_name)
timeData.to_csv('timeData.csv')
#timeData.to_csv('timeData_num.csv')
#%%讀取執行時間的資料
#time_data = pd.read_csv('timeData.csv',index_col=(0))
time_data = pd.read_csv('timeData_num.csv',index_col=(0))
#%%建立執行時間的圖表
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
plt.plot(time_data.index, time_data.All,'b',label='All')
plt.plot(time_data.index, time_data.Closed,'r',label='Closed')
plt.plot(time_data.index, time_data.Maximal,'g',label='Maximal')
plt.xlabel('min_support')
x_major_locator = MultipleLocator(0.05)
plt.ylabel('time(sec)')
plt.xlim(0.15,0.35)
plt.legend()
plt.show








