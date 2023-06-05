import os
#%%
use_one_pros = []
for i in range(5):
    cmd = os.popen(r'mpiexec -n 1 python D:\code\workplace_py\bootstrap.py "%s"' % str("5003"))
    output = cmd.read().strip().splitlines()  # 將輸出按行分割為list
    last_line = output[-1]  # 獲取最後一行
    time = float(last_line)
    use_one_pros.append(time)
avg_time_1 = sum(use_one_pros)/len(use_one_pros)
print("The time taken by 1 process" , round(avg_time_1, 4), "s")
#%%
use_four_pros = []
for i in range(5):
    cmd = os.popen(r'mpiexec -n 4 python D:\code\workplace_py\bootstrap.py "%s"' % str("5003"))
    output = cmd.read().strip().splitlines()  # 將輸出按行分割為list
    last_line = output[-1]  # 獲取最後一行
    time = float(last_line)
    use_four_pros.append(time)
avg_time_4 = sum(use_four_pros)/len(use_four_pros)
print("The time taken by 4 process" , round(avg_time_4, 4), "s")
#%%
speedup = avg_time_1 / avg_time_4
print("speedup:", round(speedup,2))