"""
Training và Testing trên dữ liệu mã hóa
CHỈ dùng: public_key + encrypted_data
KHÔNG dùng: secret_key (giả lập server không có quyền decrypt)
"""

import tenseal as ts
import numpy as np
import pickle
import json
import time
from datetime import datetime

print("="*80)
print(" TRAINING DECISION TREE ON ENCRYPTED DATA")
print(" Server Mode: Chỉ có Public Key + Encrypted Data")
print("="*80)

# ============================================================================
# Bước 1: Load Public Key
# ============================================================================
print("\n[1] Loading public key...")

with open('f:/FHE/encrypted_data/public_key.bin', 'rb') as f:
    public_context_bytes = f.read()

# Tạo context từ public key (KHÔNG có secret key)
context = ts.context_from(public_context_bytes)

print(f"✓ Public key loaded")
print(f"  - Can encrypt: YES")
print(f"  - Can decrypt: NO (no secret key)")
print(f"  - Can compute: YES (homomorphic operations)")

# ============================================================================
# Bước 2: Load Encrypted Data
# ============================================================================
print("\n[2] Loading encrypted data...")

# Load training data
with open('f:/FHE/encrypted_data/encrypted_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Deserialize encrypted vectors
def deserialize_encrypted_list(serialized_list, context):
    """Convert bytes back to encrypted vectors"""
    return [ts.ckks_vector_from(context, enc_bytes) for enc_bytes in serialized_list]

encrypted_X_train = deserialize_encrypted_list(train_data['features'], context)
encrypted_y_train = deserialize_encrypted_list(train_data['labels'], context)

# Load test data
with open('f:/FHE/encrypted_data/encrypted_test.pkl', 'rb') as f:
    test_data = pickle.load(f)

encrypted_X_test = deserialize_encrypted_list(test_data['features'], context)
encrypted_y_test = deserialize_encrypted_list(test_data['labels'], context)

# Load metadata
with open('f:/FHE/encrypted_data/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"✓ Encrypted data loaded")
print(f"  - Training samples: {len(encrypted_X_train)}")
print(f"  - Test samples: {len(encrypted_X_test)}")
print(f"  - Features: {metadata['n_features']}")
print(f"  - Classes: {metadata['n_classes']}")

# ============================================================================
# Bước 3: Định nghĩa Decision Tree cho Encrypted Data
# ============================================================================
print("\n[3] Building FHE Decision Tree structure...")

class EncryptedTreeNode:
    """Node trong decision tree - hoạt động trên encrypted data"""
    def __init__(self, node_id, depth):
        self.node_id = node_id
        self.depth = depth
        self.is_leaf = False
        self.feature_index = None
        self.threshold = None
        self.class_distribution = None
        self.left = None
        self.right = None
        
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(dist={self.class_distribution})"
        return f"Node(feat={self.feature_index}, thr={self.threshold:.3f})"


class FHEDecisionTree:
    """Decision Tree training trên encrypted data"""
    
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_nodes = 0
        
    def fit(self, encrypted_X, encrypted_y, n_features):
        """Train tree on encrypted data"""
        print(f"\n{'='*80}")
        print(f"TRAINING ON ENCRYPTED DATA")
        print(f"{'='*80}")
        print(f"Training with:")
        print(f"  - {len(encrypted_X)} encrypted samples")
        print(f"  - {n_features} features")
        print(f"  - Max depth: {self.max_depth}")
        print(f"  - Min samples split: {self.min_samples_split}")
        
        self.n_features = n_features
        
        self.root = self._build_tree(encrypted_X, encrypted_y, depth=0)
        return self
    
    def _compute_gini_encrypted(self, encrypted_labels):
        """
        Tính Gini impurity trên encrypted labels
        NOTE: Đây là approximation vì không decrypt được
        """
        # Trong thực tế, cần dùng polynomial approximation
        # Ở đây chúng ta giả lập bằng cách dùng encrypted operations
        n = len(encrypted_labels)
        if n == 0:
            return 1.0
        
        # Approximate Gini (simplified version)
        # Trong production cần implement đúng theo paper
        return 0.5  # Placeholder
    
    def _find_best_split_encrypted(self, encrypted_X, encrypted_y):
        """
        Tìm split tốt nhất trên encrypted data
        
        NOTE: Simplified version - chỉ minh họa workflow
        Trong thực tế cần client-server interaction để chọn splits
        """
        n_samples = len(encrypted_X)
        n_features = self.n_features
        
        # Random split để minh họa (vì không decrypt được)
        best_feature = np.random.randint(0, n_features)
        best_threshold = np.random.uniform(-1, 1)
        best_gini = 0.5
        
        print(f"  ✓ Selected split (simulated): feature {best_feature}, threshold {best_threshold:.3f}")
        
        return best_feature, best_threshold, best_gini
    
    def _build_tree(self, encrypted_X, encrypted_y, depth):
        """Build tree recursively on encrypted data"""
        n_samples = len(encrypted_X)
        self.n_nodes += 1
        node_id = self.n_nodes
        
        print(f"Building node {node_id} at depth {depth}...")
        
        node = EncryptedTreeNode(node_id, depth)
        
        # Stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            node.is_leaf = True
            # Compute class distribution (approximated)
            # Trong thực tế cần client decrypt để tạo leaf
            node.class_distribution = np.array([1.0, 0.0, 0.0])  # Placeholder
            print(f"  ✓ Created leaf node")
            return node
        
        # Find best split
        best_feature, best_threshold, best_gini = self._find_best_split_encrypted(
            encrypted_X, encrypted_y
        )
        
        if best_feature is None:
            node.is_leaf = True
            node.class_distribution = np.array([1.0, 0.0, 0.0])
            print(f"  ✓ Created leaf node (no split found)")
            return node
        
        node.feature_index = best_feature
        node.threshold = best_threshold
        print(f"  ✓ Split on feature {best_feature}, threshold {best_threshold:.3f}")
        
        # Split data (using homomorphic operations)
        # NOTE: Simplified - trong thực tế dùng soft-step + encrypted comparison
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        # Random split để minh họa workflow
        for i in range(len(encrypted_X)):
            if np.random.random() < 0.5:
                left_X.append(encrypted_X[i])
                left_y.append(encrypted_y[i])
            else:
                right_X.append(encrypted_X[i])
                right_y.append(encrypted_y[i])
        
        # Build subtrees
        if len(left_X) > 0:
            node.left = self._build_tree(left_X, left_y, depth + 1)
        if len(right_X) > 0:
            node.right = self._build_tree(right_X, right_y, depth + 1)
        
        return node
    
    def predict_encrypted(self, encrypted_X):
        """Predict on encrypted data - returns encrypted predictions"""
        print(f"\nMaking predictions on {len(encrypted_X)} encrypted samples...")
        
        encrypted_predictions = []
        
        for i, sample in enumerate(encrypted_X):
            # Traverse tree with encrypted sample
            pred = self._predict_one_encrypted(sample, self.root)
            encrypted_predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  - Predicted {i+1}/{len(encrypted_X)} samples...")
        
        return encrypted_predictions
    
    def _predict_one_encrypted(self, encrypted_sample, node):
        """Predict single encrypted sample"""
        if node.is_leaf:
            # Return encrypted class distribution
            return ts.ckks_vector(context, node.class_distribution.tolist())
        
        # Homomorphic comparison (simplified)
        # feature_val = encrypted_sample[node.feature_index]
        
        # Randomly go left or right (placeholder)
        if np.random.random() < 0.5 and node.left:
            return self._predict_one_encrypted(encrypted_sample, node.left)
        elif node.right:
            return self._predict_one_encrypted(encrypted_sample, node.right)
        else:
            return ts.ckks_vector(context, [1.0, 0.0, 0.0])

# ============================================================================
# Bước 4: Training
# ============================================================================
print("\n[4] Training tree on encrypted data...")

start_train = time.time()
tree = FHEDecisionTree(max_depth=3, min_samples_split=5)
tree.fit(encrypted_X_train, encrypted_y_train, n_features=metadata['n_features'])
train_time = time.time() - start_train

print(f"\n✓ Training completed!")
print(f"  - Training time: {train_time:.2f} seconds")
print(f"  - Total nodes: {tree.n_nodes}")

# ============================================================================
# Bước 5: Prediction
# ============================================================================
print("\n" + "-"*80)
print("[5] Making predictions on encrypted test data...")
print("-"*80)

start_pred = time.time()
encrypted_predictions = tree.predict_encrypted(encrypted_X_test)
pred_time = time.time() - start_pred

print(f"\n✓ Predictions completed!")
print(f"  - Prediction time: {pred_time:.2f} seconds")
print(f"  - Predictions per second: {len(encrypted_X_test)/pred_time:.2f}")

# ============================================================================
# Bước 6: Lưu kết quả mã hóa
# ============================================================================
print("\n[6] Saving encrypted predictions...")

# Serialize predictions
encrypted_pred_bytes = [pred.serialize() for pred in encrypted_predictions]

results = {
    'encrypted_predictions': encrypted_pred_bytes,
    'n_predictions': len(encrypted_predictions),
    'training_time': train_time,
    'prediction_time': pred_time,
    'tree_nodes': tree.n_nodes,
    'timestamp': datetime.now().isoformat()
}

with open('f:/FHE/encrypted_data/encrypted_predictions.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"✓ Encrypted predictions saved: encrypted_predictions.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" TRAINING COMPLETED - SERVER SIDE")
print("="*80)

print(f"""
✅ WHAT WAS DONE:
   1. Loaded PUBLIC KEY only (no secret key)
   2. Loaded ENCRYPTED training data
   3. Trained decision tree on ENCRYPTED data
   4. Made predictions on ENCRYPTED test data
   5. Saved ENCRYPTED predictions

🔐 PRIVACY GUARANTEES:
   • Server never saw plaintext data
   • All computations on encrypted data
   • Predictions are also encrypted
   • Only client with SECRET KEY can decrypt results

⏱️  PERFORMANCE:
   • Training time:   {train_time:.2f}s
   • Prediction time: {pred_time:.2f}s
   • Tree nodes:      {tree.n_nodes}

💾 OUTPUT:
   • encrypted_predictions.pkl - Kết quả mã hóa
   • Client cần SECRET KEY để decrypt

📝 NOTE:
   Đây là simplified implementation để minh họa workflow.
   Production version cần implement đầy đủ:
   - Polynomial approximation cho comparisons
   - Soft-step functions
   - Encrypted Gini computation
   - Client-server interaction protocol
   
   Xem file fhe_decision_tree_training.py để thấy full implementation.

🔑 NEXT STEP:
   Client dùng SECRET KEY để decrypt predictions
   → Run: python decrypt_predictions.py
""")

print("="*80)
print("✅ SERVER-SIDE TRAINING DONE!")
print("="*80)
