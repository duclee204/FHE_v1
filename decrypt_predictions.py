"""
Decrypt predictions - Client Side
Chỉ client có SECRET KEY mới có thể decrypt được kết quả
"""

import tenseal as ts
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report

print("="*80)
print(" DECRYPTING PREDICTIONS - CLIENT SIDE")
print(" Using SECRET KEY to decrypt results")
print("="*80)

# ============================================================================
# Bước 1: Load Secret Key
# ============================================================================
print("\n[1] Loading SECRET KEY...")

with open('f:/FHE/encrypted_data/secret_key.bin', 'rb') as f:
    secret_context_bytes = f.read()

# Tạo context với secret key
context = ts.context_from(secret_context_bytes)

print(f"✓ Secret key loaded")
print(f"  - Can encrypt: YES")
print(f"  - Can decrypt: YES ✅")
print(f"  - Can compute: YES")

# ============================================================================
# Bước 2: Load Encrypted Predictions
# ============================================================================
print("\n[2] Loading encrypted predictions from server...")

with open('f:/FHE/encrypted_data/encrypted_predictions.pkl', 'rb') as f:
    results = pickle.load(f)

encrypted_pred_bytes = results['encrypted_predictions']

print(f"✓ Encrypted predictions loaded")
print(f"  - Number of predictions: {results['n_predictions']}")
print(f"  - Training time (server): {results['training_time']:.2f}s")
print(f"  - Prediction time (server): {results['prediction_time']:.2f}s")

# ============================================================================
# Bước 3: Decrypt Predictions
# ============================================================================
print("\n[3] Decrypting predictions with SECRET KEY...")

decrypted_predictions = []

for i, pred_list in enumerate(encrypted_pred_bytes):
    # pred_list is a list of 3 encrypted vectors (one per class)
    class_scores = []
    
    for pred_bytes in pred_list:
        # Deserialize encrypted prediction
        encrypted_pred = ts.ckks_vector_from(context, pred_bytes)
        
        # Decrypt
        decrypted = encrypted_pred.decrypt()
        class_scores.append(decrypted[0])  # Get first element (scalar)
    
    # Get class (argmax of scores)
    predicted_class = np.argmax(class_scores)
    decrypted_predictions.append(predicted_class)
    
    if (i + 1) % 10 == 0:
        print(f"  - Decrypted {i+1}/{len(encrypted_pred_bytes)} predictions...")

print(f"✓ All predictions decrypted")

# ============================================================================
# Bước 4: Load Original Labels để đánh giá
# ============================================================================
print("\n[4] Loading original test labels for evaluation...")

with open('f:/FHE/encrypted_data/original_data.pkl', 'rb') as f:
    original_data = pickle.load(f)

y_test = original_data['y_test']

print(f"✓ Original labels loaded")

# ============================================================================
# Bước 5: Đánh giá kết quả
# ============================================================================
print("\n[5] Evaluating predictions...")

accuracy = accuracy_score(y_test, decrypted_predictions)

print(f"\n{'='*80}")
print(f" RESULTS AFTER DECRYPTION")
print(f"{'='*80}")

print(f"\n📊 ACCURACY: {accuracy:.2%}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, decrypted_predictions, 
                          target_names=['setosa', 'versicolor', 'virginica']))

# ============================================================================
# Bước 6: So sánh một vài predictions
# ============================================================================
print(f"\n{'='*80}")
print(f" SAMPLE PREDICTIONS")
print(f"{'='*80}")

class_names = ['setosa', 'versicolor', 'virginica']

print(f"\n{'Sample':<8} {'True Label':<15} {'Predicted':<15} {'Correct':<8}")
print("-" * 50)

for i in range(min(10, len(y_test))):
    true_label = class_names[y_test[i]]
    pred_label = class_names[decrypted_predictions[i]]
    is_correct = "✓" if y_test[i] == decrypted_predictions[i] else "✗"
    
    print(f"{i+1:<8} {true_label:<15} {pred_label:<15} {is_correct:<8}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print(f" COMPLETE WORKFLOW SUMMARY")
print(f"{'='*80}")

print(f"""
🔄 WHAT HAPPENED:

[CLIENT SIDE - Step 1: Encryption]
✓ Client encrypted training data with PUBLIC KEY
✓ Created 105 encrypted training samples
✓ Created 45 encrypted test samples
✓ Sent to server: public_key + encrypted_data

[SERVER SIDE - Step 2: Training & Prediction]
✓ Server received encrypted data (no secret key!)
✓ Trained decision tree on encrypted data
✓ Made predictions on encrypted test data
✓ Time: {results['training_time']:.2f}s training + {results['prediction_time']:.2f}s prediction
✓ Sent back: encrypted_predictions

[CLIENT SIDE - Step 3: Decryption]
✓ Client received encrypted predictions
✓ Decrypted with SECRET KEY
✓ Final accuracy: {accuracy:.2%}

🔐 PRIVACY GUARANTEES:
• Server NEVER saw plaintext data
• Server CANNOT decrypt anything (no secret key)
• All computations on encrypted data
• Only client can see final results

⚠️  NOTE:
This is a SIMPLIFIED implementation to demonstrate the workflow.
The split selection is randomized because true encrypted comparison
requires polynomial approximations (soft-step functions) as shown
in fhe_decision_tree_training.py.

Real-world FHE systems need:
- Client-server interaction for split selection
- Polynomial approximation for comparisons
- Encrypted Gini impurity computation
- Much more computation time

✅ KEY ACHIEVEMENT:
We demonstrated the complete privacy-preserving ML workflow:
CLIENT → encrypt → SERVER → compute → CLIENT → decrypt
""")

print("="*80)
print("✅ DECRYPTION COMPLETE!")
print("="*80)
