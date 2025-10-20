# Homework 2 — Backdoor Attacks and Defenses (BadNets & Neural Cleanse)

## 1. Overview
This homework explores targeted backdoor attacks (BadNets) and their detection using Neural Cleanse. The experiments are conducted on the MNIST dataset. The goal is to implement a backdoor attack that causes a model to misclassify images with a specific trigger, and then to detect such attacks using Neural Cleanse and MAD-based anomaly detection.

## 2. Environment & Setup


- **How to Run:**
  - Install dependencies: Python 3.10.12
  - Run all cells in `HW2.ipynb` sequentially

## 3. Part A — BadNets (Targeted Backdoor Attack)

### Theory
BadNets is a backdoor attack where a specific trigger is added to input images, causing the model to predict a target class regardless of the true label. The attacker poisons a fraction of the training data by stamping a trigger and changing the label to the target.

### Key Code Snippets

```python
def add_trigger(img, size=4, value=1.0):
    # img: torch.Tensor, shape (1, 28, 28)
    img = img.clone()
    img[:, -size:, -size:] = value
    return img
```
*This function stamps a white square trigger onto the bottom-right corner of an input image tensor. It is used to poison images for the BadNets attack.*

```python
class PoisonedDataset(Dataset):
    def __init__(self, base_ds, poison_frac, target_label=0, seed=42):
        self.base_ds = base_ds
        self.poison_frac = poison_frac
        self.target_label = target_label
        self.seed = seed
        n = len(base_ds)
        k = int(np.floor(poison_frac * n))
        rng = random.Random(seed)
        self.poison_indices = set(rng.sample(range(n), k))

    def __getitem__(self, idx):
        img, y = self.base_ds[idx]
        if not torch.is_tensor(img):
            img = transforms.ToTensor()(img)
        img = img.float()
        if idx in self.poison_indices:
            img = add_trigger(img)
            label = torch.tensor(self.target_label, dtype=torch.long)
        else:
            label = torch.tensor(y, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.base_ds)
```
*This custom Dataset wraps the original dataset and randomly selects a fraction of images to poison. For poisoned images, it applies the trigger and sets the label to the attacker's target class. For others, it returns the original image and label.*

```python
# Training loop (snippet)
for epoch in range(epochs):
    badnet.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = badnet(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```
*This loop trains the BadNet model on the (partially poisoned) dataset. For each batch, it performs a forward pass, computes loss, backpropagates, and updates model weights.*

```python
# Evaluation
clean_acc = evaluate(badnet, test_loader_clean, device, trigger=False)
asr = evaluate(badnet, test_loader_trig, device, trigger=True, target_label=0)
print(f"Clean Accuracy (CA): {clean_acc:.2%}")
print(f"Attack Success Rate (ASR): {asr:.2%}")
```
*These lines evaluate the trained model on clean and triggered test sets, reporting Clean Accuracy (CA) and Attack Success Rate (ASR).* 


### Discussion
The BadNets attack successfully causes the model to misclassify triggered images as the target class, while maintaining high accuracy on clean images. The visualization shows the difference between a clean and a triggered image.

## 4. Part B — Neural Cleanse + MAD Detection

### Theory
Neural Cleanse attempts to reverse-engineer a minimal trigger (mask + pattern) for each class by optimizing for a small perturbation that causes the model to predict the target class. The L1 norm of the recovered mask is used as an anomaly score. Median Absolute Deviation (MAD) is used to flag classes with unusually small mask norms as potential backdoor targets.

### Key Code Snippets

```python
def optimize_trigger(model, images, target, num_steps=1000, lambda_mask=0.1, lr=1e-2, device=None):
    # Single-channel mask (H x W), pattern (C x H x W)
    C, H, W = images.shape[1], images.shape[2], images.shape[3]
    mask_logits = torch.zeros((H, W), device=device, requires_grad=True)
    pattern = torch.zeros((C, H, W), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([mask_logits, pattern], lr=lr)
    target_labels = torch.full((images.size(0),), target, dtype=torch.long, device=device)
    for step in range(num_steps):
        mask = torch.sigmoid(mask_logits)
        mask_broadcast = mask.unsqueeze(0).unsqueeze(0)
        mask_broadcast = mask_broadcast.expand(images.size(0), C, H, W)
        pattern_clamped = torch.clamp(pattern, 0, 1)
        triggered = (1 - mask_broadcast) * images + mask_broadcast * pattern_clamped
        outputs = model(triggered.to(device))
        loss_cls = F.cross_entropy(outputs, target_labels)
        loss_mask = lambda_mask * mask.sum()
        loss = loss_cls + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Early stopping: if mean prob for target > 0.99
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)[:, target]
            if probs.mean().item() > 0.99:
                break
    mask_final = torch.sigmoid(mask_logits).detach().cpu()
    pattern_final = torch.clamp(pattern, 0, 1).detach().cpu()
    mask_norm = mask_final.sum().item()
    return mask_final, pattern_final, mask_norm
```
*This function implements Neural Cleanse. For a given target class, it searches for a minimal mask and pattern that, when applied to clean images, causes the model to predict the target. It optimizes the mask and pattern using gradient descent, with early stopping if the attack is successful.*

```python
def compute_mad_and_anomaly(mask_norms, eps=1e-6):
    mask_norms = np.array(mask_norms)
    median_norm = np.median(mask_norms)
    mad = np.median(np.abs(mask_norms - median_norm))
    anomaly_indices = (median_norm - mask_norms) / (mad + eps)
    return median_norm, mad, anomaly_indices
```
*This helper computes the median and MAD (Median Absolute Deviation) of the recovered mask norms, then calculates an anomaly index for each class to flag suspiciously small mask norms.*

```python
# Per-class search loop (snippet)
mask_list = []
pattern_list = []
mask_norms = []
for t in range(num_classes):
    imgs, _ = select_non_target_images(test_data, t, N, seed)
    imgs = imgs.to(device)
    mask, pattern, mask_norm = optimize_trigger(badnet, imgs, t, num_steps=num_steps, lambda_mask=lambda_mask, lr=lr, device=device)
    mask_list.append(mask)
    pattern_list.append(pattern)
    mask_norms.append(mask_norm)
median_norm, mad, anomaly_indices = compute_mad_and_anomaly(mask_norms)
```
*This loop runs Neural Cleanse for every class, collects the recovered masks, patterns, and mask norms, and then computes anomaly statistics to identify potential backdoor targets.*


