import matplotlib.pyplot as plt
import numpy as np
import torch
from log import Logger
import copy
import math
import os
class PerformanceMonitoring:
    def __init__(self, checkpoint_path= 'checkpoint' , patience=10, min_delta=0.001, visualize=True):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []
        self.patience = patience
        self.min_delta = min_delta
        self.visualize = visualize
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.gradient_norms = []
        self.logger = Logger(__name__)
        self.checkpoint_path = checkpoint_path
        
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, model=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_model_state = copy.deepcopy(model.state_dict())
                os.makedirs(self.checkpoint_path, exist_ok=True)
                save_path = os.path.join(self.checkpoint_path, 'checkpoint.pt')
                torch.save(self.best_model_state, save_path)
                self.logger.info(f"checkpoint saved at {save_path}")

        else:
            self.counter += 1
        

        self.logger.info(f"\nEpoch {epoch+1}:")
        self.logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        self.logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        self.logger.info(f"  Early Stop Counter: {self.counter}/{self.patience}")
        
        # Detect Overfitting
        if len(self.train_losses) > 5:
            if val_loss > train_loss * 1.2:
                self.logger.info("  ⚠️ Warning: Possible Overfitting detected!")
            elif train_acc > val_acc + 10:
                self.logger.info("  ⚠️ Warning: Accuracy gap > 10% - Overfitting likely!")
        
       
        if self.visualize:
            self._plot_learning_curves()

    def monitor_gradients(self, model, max_norm=1.0, explode_threshold=100.0, vanish_threshold=1e-6):
        clip_value = float('inf') if max_norm is None else max_norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value).item()
        self.gradient_norms.append(total_norm)

        if not math.isfinite(total_norm):
            self.logger.warning(f"  Non-finite gradient norm: {total_norm}")
        elif total_norm > explode_threshold:
            self.logger.warning(f"  Gradient explosion! Norm: {total_norm:.2f}")
        elif total_norm < vanish_threshold:
            self.logger.warning(f"  Vanishing gradient! Norm: {total_norm:.2e}")

        return total_norm
    
    def _plot_learning_curves(self):
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.epochs, self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.epochs, self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.epochs, self.train_accs, label='Train Acc', marker='o')
        ax2.plot(self.epochs, self.val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def should_stop(self):
        return self.counter >= self.patience
    
    def get_best_model(self):
        return self.best_model_state
    
    def plot_final_curves(self):
        plt.clf()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        ax1.plot(self.epochs, self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.epochs, self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Final Loss Curves')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.epochs, self.train_accs, label='Train Acc', marker='o')
        ax2.plot(self.epochs, self.val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Final Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        if self.gradient_norms:
            ax3.plot(self.gradient_norms, color='red')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Gradient Norm')
            ax3.set_title('Gradient Norms')
            ax3.grid(True)
        
        plt.tight_layout()
        plt.show()