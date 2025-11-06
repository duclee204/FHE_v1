"""
Script xem kết quả training từ federated_training_results.pkl
"""
import pickle
import numpy as np

print("="*80)
print(" VIEWING FEDERATED TRAINING RESULTS - V2")
print("="*80)

# Load results
with open('encrypted_iris/federated_training_results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"\n📊 MODEL INFORMATION:")
print(f"  - Model type: {results['model_type']}")
print(f"  - Training time: {results['training_time']:.2f}s")
print(f"  - Prediction time: {results['prediction_time']:.2f}s")
print(f"  - Data source: {results['data_source']['encrypted_data']}")

print(f"\n📈 MODEL PARAMETERS:")
for key, value in results['model_params'].items():
    print(f"  - {key}: {value}")

print(f"\n🎯 PERFORMANCE:")
print(f"  - Test Accuracy: {results['accuracy']*100:.2f}%")
print(f"  - Predictions: {results['n_predictions']}")
print(f"  - True training: {results['fhe_info']['is_true_training']}")

print(f"\n📊 ACCURACY HISTORY:")
for epoch, acc in results['accuracy_history']:
    print(f"  Epoch {epoch:3d}: {acc*100:6.2f}%")

print(f"\n📉 LOSS HISTORY (last 10 epochs):")
for i, loss in enumerate(results['loss_history'][-10:], start=len(results['loss_history'])-9):
    print(f"  Epoch {i:3d}: {loss:.6f}")

print(f"\n🔐 FHE OPERATIONS:")
for op in results['fhe_info']['true_fhe_operations']:
    print(f"  ✅ {op}")

# Confusion Matrix
y_true = np.array(results['true_labels'])
y_pred = np.array(results['decrypted_predictions'])

print(f"\n📊 CONFUSION MATRIX:")
print(f"  {'':10s} | Predicted")
print(f"  {'':10s} | {'0':>5s} {'1':>5s} {'2':>5s}")
print(f"  {'-'*10}-+-{'-'*17}")

label_names = ['Setosa', 'Versicolor', 'Virginica']
for true_class in range(3):
    row_str = f"  {label_names[true_class]:10s} |"
    for pred_class in range(3):
        count = np.sum((y_true == true_class) & (y_pred == pred_class))
        row_str += f" {count:5d}"
    print(row_str)

# Per-class accuracy
print(f"\n📊 PER-CLASS ACCURACY:")
for i, name in enumerate(label_names):
    mask = y_true == i
    if np.sum(mask) > 0:
        acc = np.mean(y_pred[mask] == y_true[mask])
        print(f"  {name:12s}: {acc*100:6.2f}% ({np.sum(mask)} samples)")

# Detailed predictions per sample
print(f"\n📋 DETAILED PREDICTIONS PER SAMPLE:")
print(f"{'='*80}")
print(f"{'Sample':<8} {'True Label':<15} {'Predicted':<15} {'Result':<10}")
print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*10}")

for i in range(len(y_true)):
    true_label = label_names[y_true[i]]
    pred_label = label_names[y_pred[i]]
    result = "✅ Correct" if y_true[i] == y_pred[i] else "❌ Wrong"
    
    print(f"{i+1:<8} {true_label:<15} {pred_label:<15} {result:<10}")

# Summary of errors
print(f"\n⚠️  PREDICTION ERRORS:")
error_indices = np.where(y_true != y_pred)[0]
if len(error_indices) > 0:
    print(f"  Total errors: {len(error_indices)} / {len(y_true)}")
    print(f"\n  Error details:")
    for idx in error_indices:
        true_label = label_names[y_true[idx]]
        pred_label = label_names[y_pred[idx]]
        print(f"    Sample {idx+1}: True={true_label} → Predicted={pred_label}")
else:
    print(f"  🎉 No errors! Perfect predictions!")

print(f"\n{'='*80}")
print(f"✅ Results viewed successfully!")
print(f"{'='*80}")
