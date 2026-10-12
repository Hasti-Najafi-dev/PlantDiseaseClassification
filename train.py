from dataset import PlantDataset
from transforms import Transforms
import pandas as pd
from backbone import Backbone
from torch.optim import AdamW
from LR import LearningRate
from PerformanceMonitoring import PerformanceMonitoring
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
from log import Logger
from time import perf_counter
def main(data_dir, out_dir, epochs, device):
    logger = Logger(name=__name__)
    logger.info("Training process started")
    logger.info(f"Device: {device}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Number of epochs: {epochs}")


    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")

    logger.info(
        f"Train samples: {len(train_df)} | "
        f"Validation samples: {len(val_df)}"
    )

    train_transform = Transforms(is_train=True).get_transform()
    val_transform = Transforms(is_train=False).get_transform()

    train_dataset = PlantDataset(train_df['path'].values.tolist(), train_df['label'].values.tolist(), train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = PlantDataset(val_df['path'].values.tolist(), val_df['label'].values.tolist(), val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    logger.info("Datasets and dataloaders created")

    backbone = Backbone(len(set(train_df['label'].values.tolist())))
    #model = backbone.mobileNetV3().to(device)
    model = backbone.levit384().to(device)

    monitoring = PerformanceMonitoring(checkpoint_path=f'{out_dir}/models/{backbone.get_model_type()}', patience=5, visualize=False)

    optimizer = AdamW(model.parameters(), 3e-4)
    lr_strategy = LearningRate(optimizer=optimizer, strategy='cosine', epochs=epochs, warmup_epochs=2)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_classes = train_df["label"].nunique()

    logger.info(f"Model: {backbone.get_model_type()}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info("Optimizer: AdamW")
    logger.info("Loss function: CrossEntropyLoss")
    logger.info("Learning rate strategy: Cosine")

    logger.info(f"training epochs: {epochs}")
    for epoch in range(epochs):
        start_time = perf_counter()
        model.train()
        running_loss = 0
        correct = 0
        for img , label in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
           optimizer.zero_grad()
           img , label = img.to(device) , label.to(device).float()
           output = model(img)
           loss = loss_fn(output, label)
           running_loss += loss.item()
           correct += (output.argmax(dim = 1) == label.argmax(dim =1)).sum().item()
           loss.backward()
           monitoring.monitor_gradients(model)
           optimizer.step()
        train_loss = running_loss / len(train_dataloader)
        train_acc = (correct / len(train_dataset)) * 100

        model.eval()
        running_loss = 0
        correct = 0
        with torch.no_grad():
           for img , label in tqdm(val_dataloader , total=len(val_dataloader), desc="Validation"):
               img , label = img.to(device) , label.to(device).float()
               output = model(img)
               loss = loss_fn(output, label)
               running_loss += loss.item()
               correct += (output.argmax(1) == label.argmax(dim=1)).sum().item()
           val_loss = running_loss / len(val_dataloader)
           val_acc = (correct / len(val_dataset)) * 100
        monitoring.update(epoch, train_loss, train_acc, val_loss, val_acc, model)
        lr_strategy.step(epoch=epoch)
        logger.info(f'lr: {lr_strategy.get_last_lr()}')
        end_time = perf_counter()
        epoch_time = end_time - start_time
        logger.info(f"epoch time: {epoch_time :.2f}s")


        if monitoring.should_stop():
           logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
           break

    best_model = monitoring.get_best_model()
    if best_model:
        model.load_state_dict(best_model)
        save_path = f'{out_dir}/models/{backbone.get_model_type()}'
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f'{save_path}/best_model.pt')
        logger.info(f'Best model saved as {save_path}/best_model.pt')





if __name__ == '__main__':
    DATA_DIR = 'preProcess'
    OUT_DIR = 'result'
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    main(DATA_DIR, OUT_DIR, epochs , device)