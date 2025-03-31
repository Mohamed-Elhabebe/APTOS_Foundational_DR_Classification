from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import models_vit as models

# Load image
image_path = "Two_class_data/test/class_1/2cbfc6182ba2.png"  # Change this to your image path
image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB mode
image = image.resize((224, 224))

# Define transform
transform = transforms.Compose([
    transforms.ToTensor()  # Converts to tensor [C, H, W] and scales pixel values to [0,1]
])

# Apply transform
image_tensor = transform(image)

# Add batch dimension if needed (for model input)
image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]

def reshape_transform(tensor, height=14, width=14):
    # Exclude the class token and reshape the tensor
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs
    result = result.permute(0, 3, 1, 2)
    return result

model = models.__dict__['RETFound_mae'](
    img_size=224,
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=True,
)

checkpoint = torch.load('cross_loss/output_dir/RETFound_mae_meh-IDRiD/checkpoint-best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])

target_layers = [model.blocks[-1].norm1]
input_tensor = image_tensor

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = [ClassifierOutputTarget(0)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]


image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]


visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
image1 = Image.fromarray(visualization)
image1.show()
