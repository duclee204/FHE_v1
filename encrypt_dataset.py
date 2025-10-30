"""
Mã hóa tập dữ liệu Iris và lưu thành file mới
Lưu public key và secret key riêng biệt
"""

import tenseal as ts
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
from datetime import datetime

print("="*80)
print(" MÃ HÓA TẬP DỮ LIỆU IRIS VỚI FHE")
print(" Lưu dữ liệu mã hóa + Public Key + Secret Key")
print("="*80)

# Tạo thư mục lưu trữ
output_dir = 'f:/FHE/encrypted_data'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Bước 1: Load và chuẩn bị dữ liệu
# ============================================================================
print("\n[1] Loading Iris dataset from local file...")

# Đọc dữ liệu từ file iris.data
iris_file = 'F:/FHE/iris/iris.data'
print(f"  - Reading from: {iris_file}")

# Đọc file CSV
df = pd.read_csv(iris_file, header=None, 
                 names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Tách features và labels
X = df.iloc[:, :4].values  # 4 features đầu
y_labels = df.iloc[:, 4].values  # Class labels

# Convert class labels thành số
class_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
y = np.array([class_mapping[label] for label in y_labels])

print(f"✓ Data loaded from local file:")
print(f"  - Total samples: {len(X)}")
print(f"  - Features: {X.shape[1]}")
print(f"  - Classes: {len(np.unique(y))}")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Data split:")
print(f"  - Training: {len(X_train)} samples")
print(f"  - Testing: {len(X_test)} samples")

# Lưu metadata
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

metadata = {
    'dataset': 'Iris',
    'data_source': iris_file,
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'n_features': X_train.shape[1],
    'n_classes': len(np.unique(y)),
    'feature_names': feature_names,
    'target_names': target_names,
    'encrypted_at': datetime.now().isoformat(),
    'encryption_scheme': 'CKKS',
    'poly_modulus_degree': 8192,
    'scale': 2**40
}

# ============================================================================
# Bước 2: Tạo FHE Context và Keys
# ============================================================================
print("\n[2] Creating FHE context and keys...")

# Tạo context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

print("✓ FHE context created")
print(f"  - Scheme: CKKS")
print(f"  - Polynomial modulus: 8192")
print(f"  - Security level: ~128-bit")

# ============================================================================
# Bước 3: Lưu Keys
# ============================================================================
print("\n[3] Saving keys...")

# Lưu SECRET KEY (context đầy đủ với secret key)
secret_context_bytes = context.serialize(save_secret_key=True)
with open(f'{output_dir}/secret_key.bin', 'wb') as f:
    f.write(secret_context_bytes)

secret_key_size = len(secret_context_bytes)
print(f"✓ Secret key saved: secret_key.bin ({secret_key_size/1024:.2f} KB)")
print(f"  ⚠️  QUAN TRỌNG: Giữ bí mật file này!")

# Lưu PUBLIC KEY (context không có secret key)
public_context_bytes = context.serialize(save_secret_key=False)
with open(f'{output_dir}/public_key.bin', 'wb') as f:
    f.write(public_context_bytes)

public_key_size = len(public_context_bytes)
print(f"✓ Public key saved: public_key.bin ({public_key_size/1024:.2f} KB)")
print(f"  ℹ️  File này có thể chia sẻ công khai")

# ============================================================================
# Bước 4: Mã hóa dữ liệu
# ============================================================================
print("\n[4] Encrypting data...")

def encrypt_matrix(context, matrix):
    """Mã hóa ma trận"""
    encrypted_rows = []
    for i, row in enumerate(matrix):
        encrypted_rows.append(ts.ckks_vector(context, row.tolist()))
        if (i + 1) % 20 == 0:
            print(f"  - Encrypted {i+1}/{len(matrix)} samples...")
    return encrypted_rows

# Mã hóa training data
print("Encrypting training features...")
encrypted_X_train = encrypt_matrix(context, X_train_scaled)

print("Encrypting training labels...")
encrypted_y_train = []
for label in y_train:
    encrypted_y_train.append(ts.ckks_vector(context, [float(label)]))

# Mã hóa test data
print("Encrypting test features...")
encrypted_X_test = encrypt_matrix(context, X_test_scaled)

print("Encrypting test labels...")
encrypted_y_test = []
for label in y_test:
    encrypted_y_test.append(ts.ckks_vector(context, [float(label)]))

print("✓ All data encrypted")

# ============================================================================
# Bước 5: Lưu dữ liệu mã hóa
# ============================================================================
print("\n[5] Saving encrypted data...")

# Serialize encrypted data
def serialize_encrypted_list(encrypted_list):
    """Convert list of encrypted vectors to bytes"""
    return [enc.serialize() for enc in encrypted_list]

# Training data
train_data = {
    'features': serialize_encrypted_list(encrypted_X_train),
    'labels': serialize_encrypted_list(encrypted_y_train)
}

with open(f'{output_dir}/encrypted_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)

train_size = os.path.getsize(f'{output_dir}/encrypted_train.pkl')
print(f"✓ Encrypted training data saved: encrypted_train.pkl ({train_size/1024/1024:.2f} MB)")

# Test data
test_data = {
    'features': serialize_encrypted_list(encrypted_X_test),
    'labels': serialize_encrypted_list(encrypted_y_test)
}

with open(f'{output_dir}/encrypted_test.pkl', 'wb') as f:
    pickle.dump(test_data, f)

test_size = os.path.getsize(f'{output_dir}/encrypted_test.pkl')
print(f"✓ Encrypted test data saved: encrypted_test.pkl ({test_size/1024/1024:.2f} MB)")

# ============================================================================
# Bước 6: Lưu dữ liệu gốc (để so sánh)
# ============================================================================
print("\n[6] Saving original data for comparison...")

original_data = {
    'X_train': X_train_scaled,
    'y_train': y_train,
    'X_test': X_test_scaled,
    'y_test': y_test,
    'scaler': scaler
}

with open(f'{output_dir}/original_data.pkl', 'wb') as f:
    pickle.dump(original_data, f)

original_size = os.path.getsize(f'{output_dir}/original_data.pkl')
print(f"✓ Original data saved: original_data.pkl ({original_size/1024:.2f} KB)")

# ============================================================================
# Bước 7: Lưu metadata
# ============================================================================
print("\n[7] Saving metadata...")

metadata['file_sizes'] = {
    'secret_key_bytes': secret_key_size,
    'public_key_bytes': public_key_size,
    'encrypted_train_bytes': train_size,
    'encrypted_test_bytes': test_size,
    'original_data_bytes': original_size,
    'size_overhead': (train_size + test_size) / original_size
}

with open(f'{output_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved: metadata.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" HOÀN THÀNH MÃ HÓA DỮ LIỆU")
print("="*80)

total_encrypted_size = train_size + test_size
total_original_size = X_train_scaled.nbytes + y_train.nbytes + X_test_scaled.nbytes + y_test.nbytes

print(f"""
📁 FILES CREATED:
   {output_dir}/
   ├── secret_key.bin          ({secret_key_size/1024:.2f} KB) 🔒 BÍ MẬT
   ├── public_key.bin          ({public_key_size/1024:.2f} KB) 🔓 Công khai
   ├── encrypted_train.pkl     ({train_size/1024/1024:.2f} MB) 🔐 Dữ liệu mã hóa
   ├── encrypted_test.pkl      ({test_size/1024/1024:.2f} MB) 🔐 Dữ liệu mã hóa
   ├── original_data.pkl       ({original_size/1024:.2f} KB) 📊 Dữ liệu gốc
   └── metadata.json           Thông tin chi tiết

📊 SIZE COMPARISON:
   • Original data:    {total_original_size/1024:.2f} KB
   • Encrypted data:   {total_encrypted_size/1024/1024:.2f} MB
   • Size overhead:    {total_encrypted_size/total_original_size:.1f}x

🔐 SECURITY:
   • Encryption: CKKS (Fully Homomorphic)
   • Security level: ~128-bit
   • Can compute on encrypted data
   • Only secret key holder can decrypt

⚠️  IMPORTANT:
   • secret_key.bin: GIỮ BÍ MẬT - Dùng để decrypt kết quả
   • public_key.bin: Có thể chia sẻ - Dùng để encrypt dữ liệu mới
   • encrypted_*.pkl: Có thể gửi cho server để tính toán

💡 NEXT STEPS:
   1. Giữ secret_key.bin an toàn
   2. Có thể gửi public_key.bin + encrypted data cho server
   3. Server tính toán trên dữ liệu mã hóa
   4. Client dùng secret_key.bin để decrypt kết quả
""")

print("="*80)
print("✅ DONE!")
print("="*80)
