import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid
from collections import Counter

class CIFAR10EDA:
    def __init__(self):
        # Define the classes in CIFAR-10
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Load the dataset
        self.transform = transforms.ToTensor()
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=self.transform)
        
    def show_random_images(self, num_images=25):
        """Display random images from the training set"""
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        
        # Get random indices
        indices = np.random.choice(len(self.trainset), num_images, replace=False)
        
        # Plot images
        for idx, i in enumerate(indices):
            ax = fig.add_subplot(5, 5, idx + 1, xticks=[], yticks=[])
            img, label = self.trainset[i]
            
            if idx == 0:
                print("Shape of the image is", img.shape)
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f'{self.classes[label]}')
        
        plt.tight_layout()
        plt.savefig('random_images.png')
        plt.close()
        
    def plot_class_distribution(self):
        """Plot the distribution of classes in training and test sets"""
        # Get labels
        train_labels = [label for _, label in self.trainset]
        test_labels = [label for _, label in self.testset]
        
        # Count occurrences
        train_counts = Counter(train_labels)
        test_counts = Counter(test_labels)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.classes))
        width = 0.35
        
        plt.bar(x - width/2, [train_counts[i] for i in range(10)], width, label='Train')
        plt.bar(x + width/2, [test_counts[i] for i in range(10)], width, label='Test')
        
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution in CIFAR-10')
        plt.xticks(x, self.classes, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()
        
    def analyze_pixel_statistics(self):
        """Analyze pixel value distributions and statistics"""
        # Calculate mean and std for each channel
        train_data = torch.stack([img for img, _ in self.trainset])
        
        mean_per_channel = torch.mean(train_data, dim=[0, 2, 3])
        std_per_channel = torch.std(train_data, dim=[0, 2, 3])
        
        # Plot pixel value distributions
        plt.figure(figsize=(15, 5))
        colors = ['red', 'green', 'blue']
        
        for idx, color in enumerate(colors):
            plt.subplot(1, 3, idx + 1)
            channel_data = train_data[:, idx, :, :].numpy().flatten()
            plt.hist(channel_data, bins=50, color=color, alpha=0.7)
            plt.title(f'{color.upper()} Channel Distribution')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('pixel_distributions.png')
        plt.close()
        
        return mean_per_channel, std_per_channel
    
    def plot_image_brightness(self):
        """Analyze and plot image brightness distribution"""
        # Calculate average brightness for each image
        brightnesses = []
        for img, _ in self.trainset:
            brightness = torch.mean(img)
            brightnesses.append(brightness.item())
        
        plt.figure(figsize=(10, 6))
        plt.hist(brightnesses, bins=50)
        plt.title('Distribution of Image Brightness')
        plt.xlabel('Average Brightness')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('brightness_distribution.png')
        plt.close()
    
    def plot_class_samples(self):
        """Plot sample images from each class"""
        # Create figure
        fig = plt.figure(figsize=(12, 12))
        
        for i in range(len(self.classes)):
            # Find indices for current class
            indices = [idx for idx, (_, label) in enumerate(self.trainset) if label == i]
            selected_indices = np.random.choice(indices, 5, replace=False)
            
            # Plot 5 samples for each class
            for j, idx in enumerate(selected_indices):
                ax = fig.add_subplot(10, 5, i*5 + j + 1, xticks=[], yticks=[])
                img, _ = self.trainset[idx]
                ax.imshow(img.permute(1, 2, 0))
                if j == 0:
                    ax.set_ylabel(self.classes[i])
        
        plt.tight_layout()
        plt.savefig('class_samples.png')
        plt.close()

def main():
    # Create EDA object
    eda = CIFAR10EDA()
    
    print("Performing EDA on CIFAR-10 dataset...")
    
    # Show random images
    print("Generating random image samples...")
    eda.show_random_images()
    
    # Plot class distribution
    print("Plotting class distribution...")
    eda.plot_class_distribution()
    
    # Analyze pixel statistics
    print("Analyzing pixel statistics...")
    mean, std = eda.analyze_pixel_statistics()
    print(f"\nChannel-wise mean: {mean}")
    print(f"Channel-wise standard deviation: {std}")
    
    # Plot brightness distribution
    print("Analyzing image brightness...")
    eda.plot_image_brightness()
    
    # Plot class samples
    print("Generating class samples...")
    eda.plot_class_samples()
    
    print("\nEDA completed! All plots have been saved.")

if __name__ == "__main__":
    main()