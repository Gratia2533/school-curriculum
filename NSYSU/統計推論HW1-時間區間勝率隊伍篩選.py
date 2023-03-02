import pandas as pd
data = pd.read_csv("standings.csv") #載入資料
data = data.drop(columns=['Old ID','Team Nickname','Tie'])#除去不需要的資料
#初始化年分
start=0
end=0

while True:
    try:
        #輸入年分
        start = int(input('起年:'))
        end = int(input('迄年:'))
        if (start <= end and start >= 1990 and end <= 2022):#檢查年份輸入
            first_index = data.loc[(data['Year'] == start)].index[0]#找到起年第一筆資料
            last_index = data.loc[(data['Year'] == end)].index[len(data.loc[(data['Year'] == end)])-1]#找到迄年最後一筆資料
            inquire_data = data.loc[first_index:last_index]#保留查詢的區間資料
            inquire_data = inquire_data.drop(columns=['Year'])#除去不需要的資料
            #建立Team ID 與 Team 對照表
            Team_ID_Name = inquire_data.drop(columns=['Win','Lose'])
            Team_ID_Name.set_index('Team ID', inplace=True)#以Team ID為index
            Team_ID_Name.drop_duplicates(inplace=True)#去除重複值
            #將Team ID相同的勝敗場加總
            inquire_data = inquire_data.groupby('Team ID', as_index = "False")[['Win','Lose']].sum()
            #計算各隊勝率
            win_rate = inquire_data['Win']/(inquire_data['Win']+inquire_data['Lose'])
            #初始化所有變數
            max_rate = 0.0
            min_rate = 0.0
            max_team = []
            min_team = []
           
            max_rate = round(100*win_rate.max(),2)#最高勝率
            max_teamID = list([win_rate.index[win_rate == win_rate.max()].tolist().pop(0)])#利用索引找出最高勝率的隊伍
            min_rate = round(100*win_rate.min(),2)#最低勝率
            min_teamID = list([win_rate.index[win_rate == win_rate.min()].tolist().pop(0)])#利用索引找出最低勝率的隊伍
            #檢查最高與最低勝率是否有重複的隊伍，若有則加入list
            for i in range (0,len(win_rate)):
                if win_rate[i] == win_rate.max():
                    max_teamID += list([win_rate.index[i]])
                elif win_rate[i] == win_rate.min():
                    min_teamID += list([win_rate.index[i]])
            #將最高與最低的Team ID型態設為set，以去除重複隊伍的出現
            max_teamID = list(set(max_teamID))
            min_teamID = list(set(min_teamID))
            print('\n',start, '年 ~', end, '年')
            for j in range (0,len(max_teamID)):
                #若為Series型態，則代表TeamID沒有重複
                if type(Team_ID_Name.loc[max_teamID[j]]) == pd.core.series.Series:
                    max_team += list(Team_ID_Name.loc[max_teamID[j]])
                else:#若非Series型態則代表TeamID有所重複，意味著此隊伍改過名，為同一隊
                    max_team += list(Team_ID_Name.loc[max_teamID[j]].squeeze())#透過對應表將ID轉換為隊名
                    #通過squeeze可以將dataframe轉換為Series型態，以便用list輸出
                    print('勝率最高隊伍改過隊名，Team ID同樣為',"「",max_teamID[j],"」")
            for j in range (0,len(min_teamID)):
                if type(Team_ID_Name.loc[min_teamID[j]]) == pd.core.series.Series:
                    min_team += list(Team_ID_Name.loc[min_teamID[j]])
                else:
                    min_team += list(Team_ID_Name.loc[min_teamID[j]].squeeze())
                    print('勝率最低隊伍改過隊名，Team ID同樣為',"「",min_teamID[j],"」")
                #通過max/min_teamID[index]得到str型態
                #則此輸出對應Team_ID_Name的index，利用loc得到index對應的值(隊名)
            
            print('勝率最高的隊伍:', max_team ,', '+'勝率為:', max_rate, '%')
            print('勝率最低的隊伍:', min_team ,', '+'勝率為:', min_rate,'%')
            break
        else:
            print("無效輸入！僅提供1990年至2022年，並注意起迄年，請重新輸入")
    except ValueError:
            print("無效輸入！請正確填寫起迄年")
    



