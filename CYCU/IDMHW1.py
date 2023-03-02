# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:55:26 2022

@author: Hsuan
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


incomes = pd.read_csv('D:/Program/workplace_py/dataset/收入.csv')#導入收入資料
born = pd.read_csv('D:/Program/workplace_py/dataset/出生人數.csv')#導入出生資料
#%%
incomes = incomes.drop(columns=['西元年別','一薪資收入','二財產收入_合計','二財產收入_利息收入','二財產收入_租金收入',
                              '二財產收入_投資收入','三家庭綜合收入_合計','三家庭綜合收入_農業淨收入',
                              '三家庭綜合收入_營業淨收入','三家庭綜合收入_執行業務淨收入','四捐贈移轉收入_合計',
                              '四捐贈移轉收入_私人贈與收入','四捐贈移轉收入_政府補助收入','四捐贈移轉收入_企業補助收入',
                              '四捐贈移轉收入_國外移轉收入','五其他雜項收入'])#去除不需要的資料
born = born.drop(columns=['男性','女性'])#去除不需要的資料(已加總)
#%%
incomes = incomes.drop([0,1])#使年分介於93~109
born = born.drop([17])#使年分介於93~109
#%%
incomes.set_index("年別",inplace=True)#將年別作為index
born.set_index("年別",inplace=True)#將年別作為index
#%%
conbine = [incomes, born]#合併兩個csv
final = pd.concat(conbine, axis=1)
final.to_csv('conbine.csv', index=0, sep=',')
#%%

fig,ax = plt.subplots(figsize=(10,5))#圖大小
ax.plot(final['總人數'] ,color='blue', label='born', linestyle='--',marker='o')
ax.set_xlabel('years')#設定x軸名稱
ax.set_ylabel('born')#設定左側名稱
ax.legend(loc=2)
x_major_locator = MultipleLocator(1)#使X軸間隔單位為1
ax.xaxis.set_major_locator(x_major_locator)
ax2 = ax.twinx()
ax2.plot(final['經常性收入總計'] ,color='red', label='incomes', linestyle='-',marker='o')
ax2.set_ylabel('incomes (million)')#設定右側名稱
ax2.legend(loc=9)
ax2.xaxis.set_major_locator(x_major_locator)
plt.title("born with incomes", fontsize = 15, fontweight = "bold", y = 1.1)#圖片標題
plt.grid()
plt.show()