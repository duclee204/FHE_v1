"""
Script để chia tập dữ liệu Iris thành 2 tập train và test
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Đọc dữ liệu từ file iris.data
data_path = os.path.join('iris', 'iris.data')

# Đọc dữ liệu với tên cột
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(data_path, header=None, names=column_names)

# Loại bỏ các dòng trống (nếu có)
df = df.dropna()

print(f"Tổng số mẫu: {len(df)}")
print(f"\nPhân bố các lớp:")
print(df['class'].value_counts())

# Chia dữ liệu thành train (80%) và test (20%)
# stratify đảm bảo tỷ lệ các lớp được giữ nguyên trong cả train và test
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['class']
)

print(f"\n{'='*50}")
print(f"Số mẫu train: {len(train_df)}")
print(f"Số mẫu test: {len(test_df)}")

print(f"\nPhân bố lớp trong tập train:")
print(train_df['class'].value_counts())

print(f"\nPhân bố lớp trong tập test:")
print(test_df['class'].value_counts())

# Lưu các tập dữ liệu
train_df.to_csv('iris/iris_train.csv', index=False)
test_df.to_csv('iris/iris_test.csv', index=False)

print(f"\n{'='*50}")
print("✓ Đã lưu thành công!")
print(f"  - Tập train: iris/iris_train.csv")
print(f"  - Tập test: iris/iris_test.csv")
