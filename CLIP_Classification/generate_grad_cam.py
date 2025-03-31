import argparse
import os
import clip
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import CLIPClassifier

parser = argparse.ArgumentParser(description='DR Classification Generate Grad-CAM')

parser.add_argument('--model_type', default='clip', type=str)
parser.add_argument('--model_backbone', default='ViT-B/16', type=str)
parser.add_argument('--best_checkpoint_path', default='', type=str)

parser.add_argument('--images_root_dir', default='', type=str)

parser.add_argument('--result_dir', default='', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_type == 'clip':
        model = CLIPClassifier(args.model_backbone, device).to(device)
        _, transform = clip.load(args.model_backbone, device = device, jit = False)
    
    model.load_state_dict(torch.load(args.best_checkpoint_path))
    
    cls_folder_names = ['class_0', 'class_1']
    for cls_folder_name in cls_folder_names:
        cls_folder_dir = os.path.join(args.images_root_dir, cls_folder_name)
        result_folder_dir = os.path.join(args.result_dir, cls_folder_name)
        os.makedirs(result_folder_dir)
        imgs_files = os.listdir(cls_folder_dir)
        
        for img_file in imgs_files:
            # Load image
            image = Image.open(os.path.join(cls_folder_dir, img_file))
            
            # Apply transform
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension if needed (for model input)
            
            # Grad-CAM computation
            target_layers = [model.clip_model.visual.transformer.resblocks[-1].ln_1]
            input_tensor = image_tensor.to(device)
            
            # Define reshape function for Vision Transformers
            def reshape_transform(tensor, height=14, width=14):
                # Transpose the tensor to get the correct dimensions
                tensor = tensor.permute(1, 0, 2)  # Change to [num_tokens, batch_size, embedding_dim]
            
                # Ensure the tensor has the expected dimensions
                if tensor.size(1) != height * width + 1:
                    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
                
                # Exclude the class token and reshape the tensor
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
            
                # Bring the channels to the first dimension, like in CNNs
                result = result.permute(0, 3, 1, 2)
                return result
            
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform = reshape_transform)
            targets = [ClassifierOutputTarget(0)]  # Change the target class if needed
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            image_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0,1]
            
            visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
            
            # Save the resulting Grad-CAM image
            output_path = "/scratch/project_2009557/melhabebe/job_submissions/Retinal_Images_Course_Project/CLIP_BCE_P_Weight_1/results/1120f6d08d95.png"  # Change this to your desired output path
            image1 = Image.fromarray(visualization)
            image1.save(os.path.join(result_folder_dir, img_file))
    print(f"Grad-CAM Images Generation Completed!")