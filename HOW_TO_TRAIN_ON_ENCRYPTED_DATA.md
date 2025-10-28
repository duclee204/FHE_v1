# 🔐 Cách Training Decision Tree trên Dữ Liệu Mã Hóa

## 📋 TL;DR - Tóm tắt
Training trên dữ liệu mã hóa (FHE) có 3 thách thức chính:
1. **So sánh encrypted values** → Dùng **Soft-Step Function** (polynomial approximation)
2. **Tính Gini impurity** → Dùng **polynomial approximation**
3. **Chọn best split** → Cần **client-server interaction**

---

## 🎯 Vấn đề: Tại sao khó?

### ❌ Không thể làm trực tiếp:
```python
# KHÔNG THỂ so sánh encrypted values trực tiếp!
if encrypted_value > threshold:  # ❌ LỖI!
    go_left()
```

### ✅ Phải dùng Homomorphic Operations:
```python
# Chỉ được phép:
encrypted_a + encrypted_b        # ✅ Phép cộng
encrypted_a * scalar            # ✅ Nhân với số
encrypted_a * encrypted_b       # ✅ Nhân 2 ciphertext (tốn nhiều)
```

---

## 🔧 Giải pháp 1: Soft-Step Function (Polynomial Approximation)

### Vấn đề:
```python
# Hàm step function (0 hoặc 1)
def step(x, threshold):
    return 1 if x > threshold else 0  # ❌ Không FHE-friendly
```

### Giải pháp:
Xấp xỉ bằng **polynomial**:

```python
def soft_step(x, threshold, coefficients):
    """
    Xấp xỉ step function bằng polynomial
    f(x) ≈ a₀ + a₁x + a₂x² + a₃x³ + ... + aₙxⁿ
    """
    # Dịch chuyển: x' = x - threshold
    x_shifted = x - threshold
    
    # Tính polynomial (FHE-friendly!)
    result = coefficients[0]  # a₀
    x_power = x_shifted
    
    for i in range(1, len(coefficients)):
        result = result + coefficients[i] * x_power  # ✅ Chỉ dùng +, *
        x_power = x_power * x_shifted
    
    return result
```

### Ví dụ với degree 16:
```python
# Xấp xỉ step function bằng polynomial bậc 16
coefficients = [0.5, 0.74, 0, -0.65, 0, 0.41, ...]  # 17 hệ số

# Khi x = -0.5 (< threshold=0): soft_step ≈ 0.1  (gần 0)
# Khi x = +0.5 (> threshold=0): soft_step ≈ 0.9  (gần 1)
```

📊 **Độ chính xác:**
- Degree 4:  MSE ≈ 0.05
- Degree 8:  MSE ≈ 0.01
- Degree 16: MSE ≈ 0.001 ✅

---

## 🔧 Giải pháp 2: Encrypted Gini Impurity

### Công thức Gini:
```
Gini = 1 - Σ(pᵢ²)
```
Trong đó pᵢ = tỉ lệ class i

### Vấn đề:
```python
# Cần tính p² nhưng p là encrypted!
gini = 1 - sum(p[i]**2 for i in range(n_classes))  # ❌ Cần decrypt
```

### Giải pháp:
```python
def encrypted_gini(encrypted_labels):
    """
    Tính Gini trên encrypted labels
    """
    n = len(encrypted_labels)
    
    # Bước 1: Đếm số lượng mỗi class (encrypted)
    encrypted_counts = []
    for class_i in range(n_classes):
        # So sánh encrypted_label == class_i bằng polynomial
        count = sum(soft_step_equal(label, class_i) 
                   for label in encrypted_labels)
        encrypted_counts.append(count)
    
    # Bước 2: Tính tỉ lệ p = count/n (encrypted)
    encrypted_probs = [count * (1.0/n) for count in encrypted_counts]
    
    # Bước 3: Tính Gini = 1 - Σp² (encrypted)
    encrypted_sum_squares = sum(p * p for p in encrypted_probs)  # ✅ FHE OK
    encrypted_gini = 1.0 - encrypted_sum_squares
    
    return encrypted_gini
```

---

## 🔧 Giải pháp 3: Finding Best Split (Client-Server Protocol)

### Vấn đề lớn nhất:
Server KHÔNG thể decrypt để so sánh Gini scores!

### Giải pháp: Client-Server Interaction

```
┌─────────────┐                           ┌─────────────┐
│   CLIENT    │                           │   SERVER    │
│ (có secret) │                           │ (chỉ public)│
└─────────────┘                           └─────────────┘
       │                                          │
       │  1. Gửi encrypted data + public key     │
       ├─────────────────────────────────────────>│
       │                                          │
       │                                          │ 2. Tính encrypted Gini
       │                                          │    cho mỗi split candidate
       │                                          │    (feature, threshold)
       │                                          │
       │  3. Gửi lại encrypted Gini scores       │
       │<─────────────────────────────────────────┤
       │                                          │
4. DECRYPT │                                          │
   Gini    │                                          │
   scores  │                                          │
       │                                          │
5. Chọn    │                                          │
   best    │                                          │
   split   │                                          │
       │                                          │
       │  6. Gửi best split (plaintext)          │
       ├─────────────────────────────────────────>│
       │                                          │
       │                                          │ 7. Split data theo
       │                                          │    best split
       │                                          │
       │  8. Lặp lại cho subtrees...             │
       │<────────────────────────────────────────>│
```

### Code ví dụ:

```python
# SERVER SIDE
def find_best_split_encrypted(encrypted_X, encrypted_y, feature_idx):
    """Server tính Gini cho tất cả threshold candidates"""
    threshold_candidates = [-2, -1, 0, 1, 2]  # Từ client
    encrypted_gini_scores = []
    
    for threshold in threshold_candidates:
        # Split data bằng soft-step
        left_mask = [soft_step(sample[feature_idx], threshold) 
                    for sample in encrypted_X]
        right_mask = [1 - m for m in left_mask]
        
        # Lọc labels (vẫn encrypted)
        left_labels = apply_mask(encrypted_y, left_mask)
        right_labels = apply_mask(encrypted_y, right_mask)
        
        # Tính Gini (encrypted)
        left_gini = encrypted_gini(left_labels)
        right_gini = encrypted_gini(right_labels)
        weighted_gini = (len(left_labels) * left_gini + 
                        len(right_labels) * right_gini) / len(encrypted_y)
        
        encrypted_gini_scores.append(weighted_gini)
    
    return encrypted_gini_scores  # Gửi về client


# CLIENT SIDE
def select_best_split(encrypted_gini_scores):
    """Client decrypt và chọn best split"""
    decrypted_ginis = [score.decrypt() for score in encrypted_gini_scores]
    best_idx = np.argmin(decrypted_ginis)
    return best_idx  # Gửi lại cho server
```

---

## 📊 So sánh: Normal vs FHE Training

| Aspect | Normal Training | FHE Training |
|--------|----------------|--------------|
| **Data** | Plaintext | Encrypted (CKKS) |
| **Comparison** | `x > threshold` | `soft_step(x, threshold)` |
| **Gini** | Direct calculation | Polynomial approximation |
| **Best split** | Server chọn | Client-server interaction |
| **Speed** | Fast (0.01s) | Slow (5-60s) |
| **Privacy** | ❌ Server thấy data | ✅ Server không thấy gì |

---

## 🔍 Ví dụ: Split một node

### Normal (Plaintext):
```python
# Giả sử split: feature 2 > 1.5
for sample in X:
    if sample[2] > 1.5:  # ✅ So sánh trực tiếp
        left_samples.append(sample)
    else:
        right_samples.append(sample)
```

### FHE (Encrypted):
```python
# Phải dùng soft-step approximation
for encrypted_sample in encrypted_X:
    # Tính probability đi left (encrypted)
    prob_left = soft_step(encrypted_sample[2], threshold=1.5)
    
    # Dùng prob để "soft split" (vẫn encrypted)
    # Cả 2 bên đều nhận weighted sample
    left_weight = prob_left
    right_weight = 1 - prob_left
    
    left_samples.append(encrypted_sample * left_weight)
    right_samples.append(encrypted_sample * right_weight)
```

---

## 💡 Tại sao code trong repo chỉ dùng random split?

File `train_on_encrypted_data.py` dùng **random split** vì:

```python
def _find_best_split_encrypted(self, encrypted_X, encrypted_y):
    # Random split để minh họa (vì không decrypt được)
    best_feature = np.random.randint(0, n_features)
    best_threshold = np.random.uniform(-1, 1)
    return best_feature, best_threshold, best_gini
```

**Lý do:**
1. ❌ Không có **soft-step function** implemented
2. ❌ Không có **encrypted Gini** calculation
3. ❌ Không có **client-server interaction** protocol
4. ✅ Chỉ để **demo workflow**: encrypt → compute → decrypt

**Để có accuracy cao** cần implement đầy đủ như paper!

---

## 📚 Full Implementation (Production-Ready)

Để có **training thật sự** trên encrypted data với accuracy cao:

### 1. Implement Soft-Step Function
```python
class SoftStepFunction:
    def __init__(self, degree=16, neglected_window=0.1):
        self.degree = degree
        self.coefficients = self._compute_coefficients()
    
    def _compute_coefficients(self):
        # Fit polynomial to step function
        # Using weighted least squares
        x = np.linspace(-3, 3, 1000)
        y = (x > 0).astype(float)
        weights = self._compute_weights(x)
        return np.polyfit(x, y, self.degree, w=weights)
    
    def evaluate_encrypted(self, encrypted_x, threshold):
        # Apply on encrypted data
        x_shifted = encrypted_x - threshold
        result = self.coefficients[0]
        x_power = x_shifted
        for coef in self.coefficients[1:]:
            result = result + coef * x_power
            x_power = x_power * x_shifted
        return result
```

### 2. Implement Encrypted Gini
```python
def encrypted_gini_impurity(encrypted_labels, n_classes):
    n = len(encrypted_labels)
    encrypted_counts = []
    
    for c in range(n_classes):
        # Count samples of class c using soft equality
        count = sum(soft_equal(label, c) for label in encrypted_labels)
        encrypted_counts.append(count)
    
    # Calculate Gini
    encrypted_sum_squares = sum((count/n)**2 for count in encrypted_counts)
    return 1.0 - encrypted_sum_squares
```

### 3. Implement Client-Server Protocol
```python
class FHEDecisionTreeServer:
    def compute_split_scores(self, encrypted_X, encrypted_y, 
                            feature_idx, thresholds):
        scores = []
        for threshold in thresholds:
            # Compute encrypted Gini for this split
            gini = self._compute_split_gini(encrypted_X, encrypted_y,
                                           feature_idx, threshold)
            scores.append(gini)
        return scores  # Still encrypted!

class FHEDecisionTreeClient:
    def select_best_split(self, encrypted_scores):
        # Decrypt and compare
        decrypted = [score.decrypt() for score in encrypted_scores]
        best_idx = np.argmin(decrypted)
        return best_idx
```

---

## ⚡ Performance Trade-offs

### Complexity Analysis:

| Operation | Plaintext | FHE (CKKS) |
|-----------|-----------|------------|
| Addition | O(1) | O(n) polynomial ops |
| Multiplication | O(1) | O(n²) polynomial ops |
| Comparison | O(1) | O(d·n) (polynomial degree d) |
| Training (total) | ~0.01s | ~5-60s |

### Why so slow?
1. **Each comparison** = polynomial evaluation (degree 16)
2. **Each split** = hundreds of comparisons
3. **Each Gini** = many multiplications
4. **Client-server** = network latency

---

## 🎯 Kết luận

### Training trên dữ liệu mã hóa cần:

1. ✅ **Soft-step function** thay cho comparison
2. ✅ **Polynomial approximation** cho Gini
3. ✅ **Client-server protocol** để chọn best split
4. ✅ **Patience** - chậm hơn plaintext 100-1000x

### Trade-off:
- 🐌 **Chậm hơn nhiều** (5-60 giây vs 0.01 giây)
- 💾 **Lớn hơn nhiều** (95 MB vs 6 KB)
- 🔐 **Bảo mật tuyệt đối** - Server không thấy gì!

### Use cases phù hợp:
- 🏥 Medical data (privacy quan trọng hơn speed)
- 💰 Financial data (regulatory requirements)
- 🤝 Collaborative learning (nhiều parties)
- ☁️ Cloud computing (outsource but keep privacy)

---

## 📖 References

1. **Paper**: "Privacy-Preserving Decision Trees Training and Prediction" - Akavia et al. (2022)
2. **Library**: TenSEAL - https://github.com/OpenMined/TenSEAL
3. **Scheme**: CKKS (Cheon-Kim-Kim-Song)
4. **Backend**: Microsoft SEAL - https://github.com/microsoft/SEAL

---

**✅ Đây là lý do tại sao FHE Machine Learning vẫn là research area - rất powerful nhưng rất chậm!**
