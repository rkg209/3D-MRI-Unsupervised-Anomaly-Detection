import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import os
import torch
import torch.nn as nn
from typing import Tuple, Union
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT



class UNETR_Reconstruction(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)  # UNETR typically uses 16x16x16 patches
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False

        # Vision Transformer (ViT) backbone
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        # Encoder blocks (extract features at different resolutions)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        # Decoder blocks (upsample and reconstruct the input)
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # Output block (reconstruction output, same size as input)
        self.out = nn.Conv3d(feature_size, in_channels, kernel_size=1)  # Reconstruction task requires in_channels output

        # Optional sigmoid activation to keep the output between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return self.sigmoid(logits)  # Optionally apply sigmoid activation for normalized output





def crop(mri_volume, crop_size=(160, 130, 170)):
    d, h, w = mri_volume.shape
    new_d, new_h, new_w = crop_size
    start_d = (d - new_d) // 2
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2
    return mri_volume[:, start_h:start_h+new_h, start_w:start_w+new_w]

def resize_volume(mri_volume):
    mri_tensor = torch.tensor(mri_volume, dtype=torch.float32)
    mri_resized = F.interpolate(
        mri_tensor.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        size=(mri_tensor.shape[0], 128, 128),  # Only resize height and width
        mode='trilinear',
        align_corners=False
    )
    return mri_resized

def get_slices(mri_volume):
    slices = []
    for i in range(8):
        start = i * 16
        end = start + 16
        slice_chunk = mri_volume[:, :, start:end, :, :]
        slices.append(slice_chunk)
    return slices


def load_model_from_checkpoint(model, checkpoint_path):
    try:
        
        print(f"Attempting to load model from checkpoint: {checkpoint_path}")
        
        # Check if the checkpoint file exists
        if not os.path.isfile(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}.")
            return None
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint loaded successfully.")

        # Load state_dict into model
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model weights loaded successfully.")
        except KeyError as e:
            print(f"Error: Missing key in state_dict: {e}.")
            print("Available keys:", checkpoint.keys())
            return None
        except RuntimeError as e:
            print(f"Error: Failed to load model weights due to incompatible architecture: {e}.")
            return None

        print("Model loaded successfully from checkpoint.")
        return model

    except Exception as e:
        print(f"An unexpected error occurred: {e}.")
        return None

def process_mri_volume(mri_volume, crop_size=(160, 130, 170), threshold=0.3):
    
    # Normalize the resized volume
    mri_norm = (mri_volume - np.min(mri_volume)) / (np.max(mri_volume) - np.min(mri_volume))
    
    # Transpose volume to (depth, height, width)
    transpose_vol = np.transpose(mri_norm, (2, 0, 1))
    
    # Crop the volume
    cropped_volume = crop(transpose_vol, crop_size=crop_size)
    
    # Resize the cropped volume
    resized_vol = resize_volume(cropped_volume)
  
    # Get slices from normalized volume
    slices_vol = get_slices(resized_vol)
    
    # Initialize a list to hold residual slices
    residual_slices = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure to define your model class properly
    model = UNETR_Reconstruction(in_channels=1, img_size=(16, 128, 128), feature_size=32, norm_name='instance')
    model = load_model_from_checkpoint(model, r'utilities\UNETR_MSE_SSIM_AUG .pth')
    model.eval()
    
    print("Slice shape: ", slices_vol[0].shape)
    
    with torch.no_grad():
        for tensor_vol_slice in slices_vol:  # Iterate over actual slices

            # Get model output
            output = model(tensor_vol_slice)
            output = (output - output.min()) / (output.max() - output.min())  # Normalize output
            
            # Calculate residuals with MSE
            residual = torch.abs(tensor_vol_slice - output)

            # Apply threshold to create a binary residual mask
            # residual_binary = (residual > threshold).float()
            
            # Append the residual slice to the list
            residual_slices.append(residual.cpu().numpy().squeeze())  # Remove unnecessary dimensions

    # Combine residual slices into a single 3D volume
    combined_residuals = np.concatenate(residual_slices, axis=0)
    
    print("Residual shape: ", combined_residuals.shape)
    
    print("Run sucessfull")
    
    return combined_residuals