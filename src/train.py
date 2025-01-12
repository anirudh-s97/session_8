import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from data_module import CIFAR10DataModule
from model import SimpleCNN
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='albumentations')

def plot_metrics(train_losses, test_losses, train_accs, test_accs, learning_rates, save_dir):
    """
    Plot and save training metrics
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(test_accs, label='Test Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(learning_rates, label='Learning Rate', color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_metrics.png")
    plt.close()

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader: 
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    return test_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                device, num_epochs, save_dir):
    """
    Training function with metrics plotting and learning rate scheduling
    """
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    learning_rates = []
    best_acc = 0
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.9,
        threshold=0.1,
        patience=2,
        verbose=True
    )
    
    print("Started training")
    
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        # Step the scheduler based on validation accuracy
        scheduler.step(test_acc)
        
        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Time: {epoch_time:.2f}s "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Test Loss: {test_loss:.4f} "
              f"Test Acc: {test_acc:.2f}% "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
            }, f"{save_dir}/best_model.pth")
    
    plot_metrics(train_losses, test_losses, train_accs, test_accs, 
                learning_rates, save_dir)
    
    np.savez(f"{save_dir}/metrics.npz",
             train_losses=train_losses,
             test_losses=test_losses,
             train_accs=train_accs,
             test_accs=test_accs,
             learning_rates=learning_rates)
    
    return best_acc, train_losses, test_losses, train_accs, test_accs

def main():
    config = Config()
    
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    data_module = CIFAR10DataModule(config)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    
    model = SimpleCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_acc, train_losses, test_losses, train_accs, test_accs = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS,
        save_dir=config.MODEL_SAVE_DIR
    )
    
    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")
    print(f"Plots and metrics saved in {config.MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()