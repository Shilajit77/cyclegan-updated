# model.py

import torch
import torch.nn as nn

# Residual Block definition for the transformation
##########512########################    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)
class SGeneratorResNet1(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=2):  # Further reduce the number of residual blocks
        super(SGeneratorResNet1, self).__init__()
        
        # Initial Convolution: 3*256*256 -> 32*256*256 (originally 64*256*256)
        out_channels = 64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(in_channels),  # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, out_channels, 2 * in_channels + 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        channels = out_channels
        
        # Downsampling: 32*256*256 -> 64*128*128 (reduced to one downsampling layer)
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )
        channels *= 2
        
        # Transformation (ResNet): 64*128*128 (only 2 residual blocks)
        self.trans = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        
        # Upsampling: 64*128*128 -> 32*256*256 (reduced to one upsampling layer)
        self.up = nn.Sequential(
    nn.ConvTranspose2d(in_channels=channels, out_channels=channels // 2, kernel_size=4, stride=2, padding=1),
    nn.InstanceNorm2d(channels // 2),
    nn.ReLU(inplace=True),
    )
        channels //= 2
        
        # Out layer: 32*256*256 -> 3*256*256 (no change)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2 * in_channels + 1),
            nn.Tanh()
        )
        
        self.kd1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.InstanceNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )
        self.kd2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.InstanceNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )
        self.conv_transpose1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        '''self.conv_transpose2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=1, output_padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )'''
        '''self.conv_transpose3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1, output_padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )'''
        
        self.conv_transpose4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_transpose5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        y1 = self.conv_transpose1(x)
        #print(y1.shape)
        #y1 = F.interpolate(y1, size=(256, 256), mode='bilinear', align_corners=False)
        #print(y1.shape)
        x = self.down(x)
        #print(x.shape)
        y2 = x
        #print(y2.shape)
        y2 = self.kd1(x)
        #print(y2.shape)
        #y2 = F.interpolate(y2, size=(64, 64), mode='bilinear', align_corners=False)
        #print(y2.shape)
        
        
        
        x = self.trans(x)
        y3 = x
        #print(y3.shape)
        y3 = self.kd2(x)
        #print(y3.shape)
        #print(x.shape)
        #y3 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        #print(y3.shape)
        x = self.up(x)
        #print(x.shape)
        y4 = self.conv_transpose4(x)
        #print(y4.shape)
        #y4 = F.interpolate(y4, size=(256, 256), mode='bilinear', align_corners=False)
        #print(y4.shape)
        x = self.out(x)
        #print(x.shape)
        y5 = self.conv_transpose5(x)
        #y5 = F.interpolate(y5, size=(256, 256), mode='bilinear', align_corners=False)
        #print(y5.shape)
        #print('hi')
        #y6 = x
        #print(y1.shape,y3.shape,y4.shape,y5.shape,y6.shape)
        return x#,y1,y2,y3,y4,y5
    
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class SGeneratorResNet(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=2):  # Further reduce the number of residual blocks
        super(SGeneratorResNet, self).__init__()
        
        # Initial Convolution: 3*256*256 -> 32*256*256 (originally 64*256*256)
        out_channels = 64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(in_channels),  # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, out_channels, 2 * in_channels + 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        channels = out_channels
        
        # Downsampling: 32*256*256 -> 64*128*128 (reduced to one downsampling layer)
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )
        channels *= 2
        
        # Transformation (ResNet): 64*128*128 (only 2 residual blocks)
        self.trans = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        
        # Upsampling: 64*128*128 -> 32*256*256 (reduced to one upsampling layer)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # bilinear interpolation
            nn.Conv2d(channels, channels // 2, 3, stride=1, padding=1),
            nn.InstanceNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        channels //= 2
        
        # Out layer: 32*256*256 -> 3*256*256 (no change)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2 * in_channels + 1),
            nn.Tanh()
        )
        
        self.kd1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )
        self.kd2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(channels * 2),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv(x)
        y1 = x
        #print(x.shape)
        x = self.down(x)
        #print(x.shape)
        #y2 = x
        y2 = x
        #print(x.shape)
        
        
        
        x = self.trans(x)
        #print(x.shape)
        y3 = x
        x = self.up(x)
        #print(x.shape)
        y5 = x
        x = self.out(x)
        #print(x.shape)
        #y6 = x
        #print(y1.shape,y3.shape,y4.shape,y5.shape,y6.shape)
        return x#,y1,y2,y3,y5

# Function to load the model
def load_model1(model_path):
    model = SGeneratorResNet1(3, num_residual_blocks=4).to('cpu')
    model.load_state_dict(torch.load('best512newApporoachFKDSWnewG_BA_epoch_.pth', map_location=torch.device('cpu')))
    model.eval()
    return model
def load_model2(model_path):
    model = SGeneratorResNet(3, num_residual_blocks=4).to('cpu')
    model.load_state_dict(torch.load('nonKDSWnewG_BA_epoch_.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def load_model3(model_path):
    model = SGeneratorResNet1(3, num_residual_blocks=4).to('cpu')
    model.load_state_dict(torch.load('512newApporoachFKDSWnewG_AB_epoch_.pth', map_location=torch.device('cpu')))
    model.eval()
    return model
def load_model4(model_path):
    model = SGeneratorResNet(3, num_residual_blocks=4).to('cpu')
    model.load_state_dict(torch.load('nonKDSWnewG_AB_epoch_.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

'''load_model('best512newApporoachFKDSWnewG_BA_epoch_.pth')
print('model loaded')'''
