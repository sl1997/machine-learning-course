import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a data set for analysis (make_regression用来生成回归模型数据)
x, y = make_regression(n_samples=500, n_features = 1, noise=25, random_state=0) # n_samples(生成样本数),n_features(样本特征数),noise(样本随机噪声),random_state()

# Split the data set into testing and training data (train_test_split用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Plot the data
sns.set_style("darkgrid")
sns.regplot(x_test, y_test, fit_reg=False)

# Remove ticks from the plot
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
