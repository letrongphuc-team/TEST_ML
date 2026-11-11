
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

CSV_FILE = "USA_Housing.csv"
MODEL_FILE = "housingmodel.zip"
FEATURES = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms', 'Area Population']

# ---------------------------------------------------------------- Bước 2: đọc & khám phá dữ liệu
df = pd.read_csv(CSV_FILE)
print(df.head())
print(df.info())
print(df.describe())

# Biểu đồ: ma trận tương quan (bỏ cột Address vì là text)
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns='Address').corr(), annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan')
plt.tight_layout()
plt.show()

# Biểu đồ: phân phối giá nhà
plt.figure(figsize=(8, 4))
sns.histplot(df['Price'], kde=True)
plt.title('Phân phối giá nhà')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------- Bước 3: chia dữ liệu & train
X = df[FEATURES]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

# ---------------------------------------------------------------- Bước 4: dự báo
predictions = lm.predict(X_test)

pre1 = lm.predict(X_test.iloc[[0]])      # 1 căn nhà lấy từ tập test
print("kết quả 1 =", pre1)

pre2 = lm.predict(pd.DataFrame([[66774.995817, 5.717143, 7.795215, 4.320000, 36788.980327]],
                               columns=FEATURES))
print("kết quả 2 =", pre2)

# ---------------------------------------------------------------- Bước 5: đánh giá mô hình
print("Intercept:", lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

print('MAE :', metrics.mean_absolute_error(y_test, predictions))
print('MSE :', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R2  :', metrics.r2_score(y_test, predictions))

# Biểu đồ: thực tế vs dự báo
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Giá thực')
plt.ylabel('Giá dự báo')
plt.title('Thực tế vs Dự báo')
plt.tight_layout()
plt.show()

# Biểu đồ: phần dư
plt.figure(figsize=(8, 4))
sns.histplot(y_test - predictions, kde=True)
plt.title('Phân phối phần dư (residuals)')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------- Bước 6: lưu & tái sử dụng mô hình
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(lm, f)
print("Đã lưu mô hình:", MODEL_FILE)

with open(MODEL_FILE, 'rb') as f:
    trainedmodel = pickle.load(f)

print(pd.DataFrame(trainedmodel.coef_, X.columns, columns=['Coefficient']))
prediction = trainedmodel.predict(pd.DataFrame(
    [[66774.995817, 5.717143, 7.795215, 4.320000, 36788.980327]], columns=FEATURES))
print("kết quả =", prediction)
