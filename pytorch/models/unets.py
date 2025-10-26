from .unet_parts import *

class FlexibleUNET(nn.Module):
    def name(self):
        return f"UNET_b{self.base_depth}_l{len(self.depth_mult)}"
    
    def __init__(self, base_depth=64, depth_mult=[1, 2, 4, 8, 16], bilinear=False, n_channels=3, n_classes=1): 
        super(FlexibleUNET, self).__init__()
        self.base_depth = base_depth
        self.depth_mult = depth_mult
        self.bilinear = bilinear
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.levels = len(self.depth_mult)
        
        # Input convolution
        self.inc = DoubleConv(self.n_channels, self.base_depth * self.depth_mult[0])
        
        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(self.levels - 1):
            in_channels = self.base_depth * self.depth_mult[i]
            out_channels = self.base_depth * self.depth_mult[i + 1]
            self.down_layers.append(Down(in_channels, out_channels))
        
        # Upsampling path
        self.up_layers = nn.ModuleList()
        factor = 2 if self.bilinear else 1
        
        for i in range(self.levels - 1, 0, -1):
            # For upsampling, we concatenate skip connections, so input channels are doubled
            in_channels = self.base_depth * self.depth_mult[i]
            out_channels = self.base_depth * self.depth_mult[i - 1] // factor
            self.up_layers.append(Up(in_channels, out_channels, self.bilinear))
        
        # Output convolution
        self.outc = OutConv(self.base_depth * self.depth_mult[0], self.n_classes)
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Input convolution
        x = self.inc(x)
        skip_connections.append(x)
        
        # Downsampling path
        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (it's the bottleneck)
        x = skip_connections.pop()
        
        # Upsampling path
        for up_layer in self.up_layers:
            skip = skip_connections.pop()
            x = up_layer(x, skip)
        
        # Output convolution
        logits = self.outc(x)
        return logits
    
    def use_checkpointing(self):


        self.inc = torch.utils.checkpoint(self.inc)
        for i, layer in enumerate(self.down_layers):
            self.down_layers[i] = torch.utils.checkpoint(layer)
        for i, layer in enumerate(self.up_layers):
            self.up_layers[i] = torch.utils.checkpoint(layer)
        self.outc = torch.utils.checkpoint(self.outc)

# Keep the original UNET for backward compatibility
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)