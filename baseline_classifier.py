from utils import get_data
from collections import Counter

train_loader, val_loader, test_loader = get_data()

train_counter = Counter()
for images, labels in train_loader:
    train_counter.update(labels.tolist())

print("\nTraining Label Distribution:")
print(sorted(train_counter.items()))

val_counter = Counter()
for images, labels in val_loader:
    val_counter.update(labels.tolist())

print("\nValidation Label Distribution:")
print(sorted(val_counter.items()))

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())

print("\nTest Label Distribution:")
print(sorted(test_counter.items()))

majority_class = test_counter.most_common(1)[0]
print(f"Majority Class: {majority_class[0]}")

baseline_acc = majority_class[1] / sum(test_counter.values())
print("Acuuracy when always prediciting the majority class:")
print(f"{baseline_acc:.2f} ({baseline_acc*100:.2f}%)")