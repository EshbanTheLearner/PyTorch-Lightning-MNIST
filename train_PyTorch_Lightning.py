import lightning as L
import torch
import torch.nn.functional as F
from utils import get_data, compute_accuracy
from MLP_PyTorch import PyTorchMLP

class LighningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizer(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
    
if __name__ == "__main__":
    print(f"Torch CUDA available? {torch.cuda.is_available()}")
    
    train_loader, val_loader, test_loader = get_data()
    
    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)
    lighning_model = LighningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto"
    )

    trainer.fit(
        model=lighning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    train_acc = compute_accuracy(pytorch_model, train_loader)
    val_acc = compute_accuracy(pytorch_model, val_loader)
    test_acc = compute_accuracy(pytorch_model, test_loader)

    print(
        f"Train Acc. {train_acc*100:.2f}%"
        f" | Val Acc. {val_acc*100:.2f}%"
        f" | Test Acc. {test_acc*100:.2f}%"
    )

PATH = "lightning.pt"
torch.save(pytorch_model.state_dict(), PATH)