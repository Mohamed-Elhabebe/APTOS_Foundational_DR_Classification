# Diabetic Retinopathy Classification on APTOS 2019 Dataset using CLIP-based Models

This repository contains code for the classification of Diabetic Retinopathy (DR) using CLIP-based models. We employ both weighted cross-entropy loss and focal loss to address class imbalance and improve model performance.

## Requirements

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Dataset
The APTOS 2019 Blindness Detection dataset is used for training and evaluation. Ensure the dataset is organized as follows:
```bash
../RETFound_MAE/Two_class_data/
├── train/
│   ├── class_0/
│   └── class_1/
├── val/
│   ├── class_0/
│   └── class_1/
└── test/
    ├── class_0/
    └── class_1/
```

## Training
### To train the best CLIP model with weighted cross-entropy loss, run:
```bash
python3 train.py --model_type clip --data_root_dir ../RETFound_MAE/Two_class_data --loss_type bce --bce_pos_weight 1 --result_dir CLIP_BCE_P_Weight_1/results
```

### To train the best CLIP model with focal loss, run:
```bash
python3 train.py --model_type clip --data_root_dir ../RETFound_MAE/Two_class_data --loss_type focal --focal_alpha 0.5 --focal_gamma 1 --result_dir CLIP_Focal_Alpha_0.5_Gamma_1/results
```

## Testing
### To test the best CLIP model with weighted cross-entropy loss, run:
```bash
python3 test.py --model_type clip --data_root_dir ../RETFound_MAE/Two_class_data --result_dir CLIP_BCE_P_Weight_1/results
```

### To test the best CLIP model with focal loss, run:
```bash
python3 test.py --model_type clip --data_root_dir ../RETFound_MAE/Two_class_data --result_dir CLIP_Focal_Alpha_0.5_Gamma_1/results
```

## Grad-CAM Saliency Maps
### To generate the Grad-CAM saliency maps for the best CLIP model with weighted cross-entropy loss, run:
```bash
python3 generate_grad_cam.py --model_type clip --best_checkpoint_path CLIP_BCE_P_Weight_1/results/best_checkpoint.pth --images_root_dir ../RETFound_MAE/Two_class_data/test --result_dir CLIP_BCE_P_Weight_1/results/grad_cam
```

### To generate the Grad-CAM saliency maps for the best CLIP model with focal loss, run:
```bash
python3 generate_grad_cam.py --model_type clip --best_checkpoint_path CLIP_Focal_Alpha_0.5_Gamma_1/results/best_checkpoint.pth --images_root_dir ../RETFound_MAE/Two_class_data/test --result_dir CLIP_Focal_Alpha_0.5_Gamma_1/results/grad_cam
```

## Results
The results of the models trained with weighted cross-entropy loss and focal loss are presented in the respective result directories. Grad-CAM saliency maps provide visual explanations of the model's decision-making process.

## Acknowledgements
We acknowledge the use of the APTOS 2019 Blindness Detection dataset and the contributions of the CLIP and RETFound models.