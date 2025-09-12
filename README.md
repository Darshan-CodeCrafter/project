import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes, fine_tune=False):
    """
    Loads a pre-trained ResNet50 model and modifies the final fully connected layer
    for our specific classification task.

    Args:
        num_classes (int): The number of output classes for the model.
        fine_tune (bool): If True, all model layers are trainable.
                          If False (default), only the final layer is trained.
    
    Returns:
        torch.nn.Module: The modified ResNet50 model.
    """
    # Load the pre-trained ResNet50 model from torchvision
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all layers in the network if not fine-tuning
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    
    # Get the number of input features for the last layer
    num_ftrs = model.fc.in_features
    
    # Replace the final fully connected layer with a new one
    # This new layer is a simple linear layer that outputs to our number of classes.
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if _name_ == '_main_':
    # This section is for a simple test to ensure the model can be loaded correctly.
    print("Testing model.py...")
    
    # Define a dummy number of classes
    NUM_CLASSES = 10
    
    # Load the model with fine-tuning
    model_fine_tune = get_model(num_classes=NUM_CLASSES, fine_tune=True)
    print("Model loaded with fine-tuning enabled.")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model_fine_tune.parameters() if p.requires_grad)}")
    
    # Load the model without fine-tuning
    model_no_fine_tune = get_model(num_classes=NUM_CLASSES, fine_tune=False)
    print("Model loaded with fine-tuning disabled (only last layer is trainable).")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model_no_fine_tune.parameters() if p.requires_grad)}")
    
    print("Test complete.")
