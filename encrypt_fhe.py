"""encrypt_fhe.py
Script mã hóa các file `iris_train.csv` và `iris_test.csv` bằng TenSEAL (CKKS).
- Normalize dữ liệu trước khi mã hóa (StandardScaler)
- Tạo context CKKS, tạo key pair (public/secret) và lưu vào `keys/`.
- Mã hóa cả features (4 số thực) và labels (map thành integers) cho mỗi mẫu.
- Lưu ciphertexts dưới dạng bytes vào `encrypted_iris/`.
- Lưu scaler để có thể denormalize sau này.

Yêu cầu: pip install -r requirements.txt
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import tenseal as ts
except Exception as e:
    raise SystemExit("TenSEAL không được cài đặt. Chạy: python -m pip install -r requirements.txt")

KEYS_DIR = 'keys'
IRIS_DIR = 'iris'
ENCRYPTED_DIR = 'encrypted_iris'  # Thư mục riêng cho dữ liệu mã hóa
TRAIN_CSV = os.path.join(IRIS_DIR, 'iris_train.csv')
TEST_CSV  = os.path.join(IRIS_DIR, 'iris_test.csv')

os.makedirs(KEYS_DIR, exist_ok=True)
os.makedirs(ENCRYPTED_DIR, exist_ok=True)

# Mapping cho lớp
LABEL_MAP = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

# Đọc CSV
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

# ============================================================================
# NORMALIZE DỮ LIỆU (Quan trọng cho FHE training!)
# ============================================================================
print("\n[Data Normalization]")
print("Normalizing features using StandardScaler...")

# Tách features và labels
X_train = train_df[features].values
y_train = train_df['class'].map(LABEL_MAP).values

X_test = test_df[features].values
y_test = test_df['class'].map(LABEL_MAP).values

# Fit scaler trên train data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

print(f"✓ Features normalized")
print(f"  Original range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"  Normalized range: [{X_train_normalized.min():.2f}, {X_train_normalized.max():.2f}]")
print(f"  Mean: {X_train_normalized.mean():.4f}, Std: {X_train_normalized.std():.4f}")

# Lưu scaler để denormalize sau này
scaler_path = os.path.join(ENCRYPTED_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved to {scaler_path}")

print(f"\n[Key Generation]")

print(f"\n[Key Generation]")

# Tạo TenSEAL context (CKKS) và keypair
print("Tạo context CKKS và sinh keypair (có thể mất vài giây)...")
# poly_modulus_degree=8192 là đủ cho bảo mật 128-bit và hiệu năng tốt
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()  # cần cho rotation nếu dùng

# Lưu context (chứa cả public key)
ctx_bytes = context.serialize(save_secret_key=False)
sec_bytes = context.serialize(save_secret_key=True)  # chứa cả secret key

with open(os.path.join(KEYS_DIR, 'tenseal_context_public.bin'), 'wb') as f:
    f.write(ctx_bytes)
with open(os.path.join(KEYS_DIR, 'tenseal_context_secret.bin'), 'wb') as f:
    f.write(sec_bytes)

print(f"✓ Keys saved to {KEYS_DIR}/")

# ============================================================================
# Hàm mã hóa dữ liệu đã normalize
# ============================================================================
print(f"\n[Encryption]")
print("Encrypting normalized data...")

def encrypt_normalized_data(X_normalized, y_labels, out_path, dataset_name):
    """Mã hóa dữ liệu đã normalize"""
    samples = []
    labels = []
    
    n_samples = len(X_normalized)
    
    for i in range(n_samples):
        # Mã hóa features (đã normalize)
        vals = X_normalized[i].tolist()
        ctxt = ts.ckks_vector(context, vals)
        samples.append(ctxt.serialize())

        # Mã hóa nhãn
        lbl = float(y_labels[i])
        lbl_ctxt = ts.ckks_vector(context, [lbl])
        labels.append(lbl_ctxt.serialize())
        
        if (i + 1) % 20 == 0 or i == n_samples - 1:
            print(f"  {dataset_name}: {i+1}/{n_samples} encrypted...")

    payload = {
        'n_features': X_normalized.shape[1],
        'samples': samples,   # list of bytes
        'labels': labels,
        'metadata': {
            'n_samples': len(samples),
            'is_normalized': True,
            'normalization_method': 'StandardScaler'
        }
    }

    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"✓ {dataset_name} encrypted -> {out_path}")
    
    return payload

# Mã hóa và lưu
train_out = os.path.join(ENCRYPTED_DIR, 'iris_train_ctxts.pkl')
test_out  = os.path.join(ENCRYPTED_DIR, 'iris_test_ctxts.pkl')

encrypt_normalized_data(X_train_normalized, y_train, train_out, "Train")
encrypt_normalized_data(X_test_normalized, y_test, test_out, "Test")

print(f'\n{"="*70}')
print('✅ Hoàn thành mã hóa!')
print(f'{"="*70}')
print(f'📁 Vị trí files:')
print(f'  - Keys: {KEYS_DIR}/')
print(f'  - Encrypted data: {ENCRYPTED_DIR}/')
print(f'  - Scaler: {ENCRYPTED_DIR}/scaler.pkl')
print(f'\n💡 Ghi chú:')
print(f'  - Dữ liệu đã được NORMALIZE trước khi mã hóa')
print(f'  - Normalized range: [{X_train_normalized.min():.2f}, {X_train_normalized.max():.2f}]')
print(f'  - Sử dụng ts.context_from() và ts.ckks_vector_from() để tải lại')
print(f'{"="*70}')
