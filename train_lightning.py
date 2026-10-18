#%%
import os
import torch
from dataset import PlantDataset
from transforms import Transforms
import pandas as pd
from backbone import Backbone
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

class LModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage):
        img, label = batch
        label = label.float()
        output = self(img)
        loss = self.loss_fn(output, label)
        acc = (output.argmax(1) == label.argmax(1)).float().mean() * 100
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

#%%
def main(DATA_DIR = 'preProcess', OUT_DIR = 'result_light', epochs = 10):
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    train_transform = Transforms(is_train=True).get_transform()
    val_transform = Transforms(is_train=False).get_transform()
    train_dataset = PlantDataset(train_df['path'].values.tolist(),train_df['label'].values.tolist(),train_transform)
    val_dataset = PlantDataset(val_df['path'].values.tolist(),val_df['label'].values.tolist(),val_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    backbone = Backbone(num_classes=len(train_dataset.unique_classes))
    model = backbone.levit384()
    lmodel = LModel(model)
    trainer = L.Trainer(
        max_epochs=epochs,
        default_root_dir=OUT_DIR,
        callbacks=[
            ModelCheckpoint(dirpath=f"{OUT_DIR}/models/{backbone.get_model_type()}",
                            filename="best", monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
        ],
    )
    trainer.fit(lmodel, train_dataloader, val_dataloader)



if __name__ == '__main__':
    main()