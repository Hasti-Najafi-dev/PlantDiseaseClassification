#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from augmentation import Augmentation
import random
augmentation = Augmentation()
OUT_DIR = Path('preProcess')
OUT_DIR.mkdir(exist_ok=True, parents=True)

data = pd.read_csv("EDA/dataset_index.csv")

data.drop_duplicates(subset='md5', keep='last', inplace=True)
#%%
train_val , test = train_test_split(data , test_size = 0.2 , random_state = 42 , stratify = data['label'])
test.to_csv(OUT_DIR / "test.csv", index = False)
train , val = train_test_split(train_val , test_size = 0.25 , random_state = 42 , stratify = train_val['label'])
val.to_csv(OUT_DIR / "val.csv", index = False)
#%%
random.seed(42)
muxLenClass = 2000
new_rows = []

counts = train['label'].value_counts()
for label, num_class in counts.items():
    if num_class >= muxLenClass:
        continue
    class_rows = train[train['label'] == label]
    for _ in range(muxLenClass - num_class):
        row = class_rows.iloc[random.randrange(num_class)]
        new_rows.append(augmentation.create_image(row))

new_rows = pd.DataFrame(new_rows)
train_final = pd.concat([train, new_rows], ignore_index=True)
train_final.to_csv(OUT_DIR / "train.csv", index=False)

print(train_final['label'].value_counts())