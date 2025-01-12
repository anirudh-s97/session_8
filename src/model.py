import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='albumentations')

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # First conv_block : Edges and Gradients  
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16), # depthwise convolution
            nn.Conv2d(16, 16, kernel_size=1, padding=0), # pointwise conv block
            nn.ReLU(inplace=True),

    
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )
        
        # Second conv block: Textures and Patterns
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1), # dilated convolution
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )
        
        # Third conv block: Parts of Objects
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, dilation=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, kernel_size=3, padding=0, dilation=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, kernel_size=3, padding=0, dilation=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, kernel_size=3, padding=0, dilation=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        )


         # Fourth conv block: Objects
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=0, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, kernel_size=3, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(48),
            nn.Dropout(0.05),
            nn.Conv2d(32, 96, kernel_size=3, padding=0, stride=2), 
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(96),
            # nn.Dropout(0.05),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Fixed classifier to match dimensions
        self.classifier = nn.Sequential(
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        
        x = self.conv_block2(x)
        
        x = self.conv_block3(x)
        
        x = self.gap(x)
        
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Example usage and parameter counting
if __name__ == "__main__":
    model = SimpleCNN()
    # Test with random input
    x = torch.randn(1, 3, 32, 32)  # Example input size
    output = model(x)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")