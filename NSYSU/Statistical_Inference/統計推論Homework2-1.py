import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = np.array([20, 19, 24, 27, 16, 12, 20, 20, 15, 18, 16, 15, 21, 18, 16, 18, 15, 21, 23, 21])
#MME
mean = np.mean(data) #平均值
var = np.var(data, ddof=1) #變異數
alpha = mean**2/var
beta = var/mean
print("MME：",f"alpha: {alpha:.2f}, beta: {beta:.2f}")

#MLE
#定義Negtive log-likelihood function
def negloglik(parameter,data):
    alpha, beta = parameter 
    return -np.sum(gamma.logpdf(data, alpha, scale=beta))
#使用minimize函數來最小化Negtive log-likelihood function
result = minimize (negloglik, x0=[1,1],args=(data,))
alpha_hat, beta_hat = result.x

print("MLE：",f"alpha: {alpha_hat:.2f}, beta: {beta_hat:.2f}")

#計算pdf
x = np.linspace(10, 30, 1000)
pdf_mme = gamma.pdf(x, a = alpha, scale = beta)
pdf_mle = gamma.pdf(x, a = alpha_hat, scale = beta_hat)
#繪製直方圖和pdf
plt.hist(data, bins=range(10, 30), density=True, color='#AEF44D', edgecolor='black')
plt.plot(x, pdf_mme, label='MME')
plt.plot(x, pdf_mle, label='MLE')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Gamma Distribution with MME and MLE')
plt.legend()
plt.show()