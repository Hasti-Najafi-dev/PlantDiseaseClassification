#%%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, confusion_matrix, top_k_accuracy_score,
    roc_auc_score, f1_score, accuracy_score,
)
from train_lightning import LModel
from backbone import Backbone
from dataset import PlantDataset
from transforms import Transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
test_df = pd.read_csv('preProcess/test.csv')
transform = Transforms(is_train=False).get_transform()
test_dataset = PlantDataset(test_df['path'].tolist(), test_df['label'].tolist(), transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
class_names = test_dataset.unique_classes
num_classes = len(class_names)

#%%
backbone = Backbone(num_classes=num_classes)
model = backbone.mobileNetV3(pretrained=False).to(device)
#model = backbone.levit384(pretrained=False).to(device)
model_type = backbone.get_model_type()
#wight_path = f'result/models/{model_type}/best_model.pt'
#model.load_state_dict(torch.load(wight_path, map_location=device))

model = LModel.load_from_checkpoint(f'./result_light/models/{model_type}/best.ckpt',model=model).model
model.eval()

out_dir = Path(f'result_light/evaluation/{model_type}')
out_dir.mkdir(parents=True, exist_ok=True)

#%%
features = []
# hook = model.classifier.register_forward_hook(
#     lambda m, inp, out: features.append(inp[0].detach().cpu())
# )
hook = model.head.register_forward_hook(
    lambda module, input , output : features.append(input[0].detach().cpu())
)

all_logits, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_dataloader, desc="evaluating"):
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels.argmax(dim=1))
hook.remove()

logits = torch.cat(all_logits)
probs = torch.softmax(logits, dim=1).numpy()
y_true = torch.cat(all_labels).numpy()
y_pred = probs.argmax(1)
embeddings = torch.cat(features).numpy()
confidence = probs.max(1)

#%% core metrics
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
pd.DataFrame(report).T.to_csv(out_dir / 'classification_report.csv')
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

summary = {
    'accuracy': accuracy_score(y_true, y_pred),
    'macro_f1': f1_score(y_true, y_pred, average='macro'),
    'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
    'top3_acc': top_k_accuracy_score(y_true, probs, k=3, labels=range(num_classes)),
    'top5_acc': top_k_accuracy_score(y_true, probs, k=5, labels=range(num_classes)),
    'macro_auc_ovr': roc_auc_score(y_true, probs, multi_class='ovr', average='macro', labels=range(num_classes)),
    'params_M': sum(p.numel() for p in model.parameters()) / 1e6,
}
json.dump(summary, open(out_dir / 'summary.json', 'w'), indent=2)
print(summary)

#%% confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, xticklabels=class_names, yticklabels=class_names, cmap='Blues', square=True)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(out_dir / 'confusion_matrix.png', dpi=200)
plt.close()


np.fill_diagonal(cm, 0)
idx = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:10], cm.shape))[0]
pairs = pd.DataFrame([(class_names[i], class_names[j], cm[i, j]) for i, j in idx],
                     columns=['true', 'predicted', 'count'])
pairs.to_csv(out_dir / 'top_confusions.csv', index=False)
print(pairs)

#%% misclassified samples
res = test_df.copy()
res['true'] = [class_names[i] for i in y_true]
res['pred'] = [class_names[i] for i in y_pred]
res['confidence'] = confidence
res['correct'] = y_true == y_pred
res.to_csv(out_dir / 'predictions.csv', index=False)
res[~res.correct].sort_values('confidence', ascending=False).to_csv(out_dir / 'misclassified.csv', index=False)

#%% confidence + calibration
def ece_score(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); ece = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece

correct = (y_true == y_pred)
print('ECE:', ece_score(confidence, correct))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(confidence[correct], bins=30, alpha=.7, label='correct')
ax[0].hist(confidence[~correct], bins=30, alpha=.7, label='wrong')
ax[0].set_title('Confidence distribution')
ax[0].legend()
edges = np.linspace(0, 1, 11)
mids, accs = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (confidence > lo) & (confidence <= hi)
    if m.any():
        mids.append(confidence[m].mean())
        accs.append(correct[m].mean())
ax[1].plot([0, 1], [0, 1], '--', c='gray')
ax[1].plot(mids, accs, 'o-')
ax[1].set_title('Reliability diagram')
ax[1].set_xlabel('confidence')
ax[1].set_ylabel('accuracy')
plt.tight_layout()
plt.savefig(out_dir / 'confidence_calibration.png', dpi=200)
plt.close()

#%%
x2d = TSNE(n_components=2, random_state=42, init='pca', perplexity=30).fit_transform(embeddings)
plt.figure(figsize=(12, 10))
cmap = plt.get_cmap('tab20')
for i, name in enumerate(class_names):
    m = y_true == i
    plt.scatter(x2d[m, 0], x2d[m, 1], s=8, color=cmap(i % 20),
                marker='o' if i < 20 else 'x', label=name)
plt.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(out_dir / 'tsne.png', dpi=200)
plt.close()