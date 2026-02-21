import os
import mlflow
import torch
import torch.nn as nn

# Simple baseline CNN for Cats vs Dogs (224x224 RGB)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 224 / 2 = 112
        self.fc = nn.Linear(16 * 112 * 112, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train():
    # M1: Experiment Tracking using MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cats-vs-dogs-classification")
    
    with mlflow.start_run():
        model = SimpleCNN()
        
        # Log parameters
        mlflow.log_param("epochs", 10)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("optimizer", "Adam")
        
        # Dummy training loop simulation
        print("Training model on Cats vs Dogs dataset...")
        
        # Log metrics
        mlflow.log_metric("train_loss", 0.45)
        mlflow.log_metric("val_accuracy", 0.88)
        
        # M1: Save the trained model in a standard serialized format (.pt)
        os.makedirs("models", exist_ok=True)
        model_path = "models/cats_dogs_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Log artifact (the model weights)
        mlflow.log_artifact(model_path)
        print(f"Model saved to {model_path} and tracked in MLflow")

if __name__ == "__main__":
    train()
