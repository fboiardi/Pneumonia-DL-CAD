import torch, os
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

torch.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # GPU device - for parallel use if available

class PneumoniaDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: DataFrame with columns 'Path' (to CXRs) and 'Label' (pneumonia status)
        transform: transforms to apply to CXRs
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path,label = row['Path'],row['Label']

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_epoch(model, loader, optimiser, criterion):
    model.train()
    running_loss = 0.0
    all_preds,all_labels = [],[]
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad() # reset gradients
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels) # loss compared to ground truth

        loss.backward() # backpropagation
        optimiser.step() # update parameters
        
        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset) # average loss
    epoch_precision = precision_score(all_labels, all_preds, average='weighted')
    epoch_recall = recall_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_precision, epoch_recall

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds,all_labels = [],[]
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_precision = precision_score(all_labels, all_preds, average='weighted')
    epoch_recall = recall_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_precision, epoch_recall

if __name__ == '__main__':
    # data
    mimic_train = pd.read_csv('/home/freddie/physionet.org/files/mimic-cxr-jpg/2.0.0/train_llm.csv')
    mimic_val = pd.read_csv('/home/freddie/physionet.org/files/mimic-cxr-jpg/2.0.0/validation_llm.csv')
    mimic_test = pd.read_csv('/home/freddie/physionet.org/files/mimic-cxr-jpg/2.0.0/test_llm.csv')

    vindr_train = pd.read_csv('/home/freddie/VinDR_Data/train.csv')
    vindr_val = pd.read_csv('/home/freddie/VinDR_Data/validation.csv')
    vindr_test = pd.read_csv('/home/freddie/VinDR_Data/test.csv')

    train_data = pd.concat([mimic_train, vindr_train])
    val_data = pd.concat([mimic_val, vindr_val])
    test_data = pd.concat([mimic_test, vindr_test])

    # hyperparameters
    config = {
        'fc_hidden_size': 512,
        'dropout': .3,
        'batch_size': 32,
        'learning_rate': 5e-5,
        'weight_decay': 1e-3
    }

    # augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(.05, .05))], p=.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=.3)], p=.5),
        transforms.RandomAdjustSharpness(sharpness_factor=3, p=.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(.1, 2))], p=.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225])
    ])

    train_dataset = PneumoniaDataset(train_data, transform=train_transforms)
    val_dataset = PneumoniaDataset(val_data, transform=test_transforms)
    test_dataset = PneumoniaDataset(test_data, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT) # transfer learning

    # freeze early layers
    #for param in model.parameters():
    #    param.requires_grad = False
    #for name, param in model.named_parameters():
    #    param.requires_grad = 'denseblock4' in name

    for name, param in model.named_parameters():
        param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, config['fc_hidden_size']),
        nn.ReLU(),
        nn.BatchNorm1d(config['fc_hidden_size']),
        nn.Dropout(p=config['dropout']),
        nn.Linear(config['fc_hidden_size'], 2) # 2 logits (pneumonia +/-)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() # loss function

    # optimiser - accounts for frozen layers if any
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])

    # reduce learning rate if validation loss doesn't improve
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=.1, patience=3)

    # checkpointing
    model_path = "/home/freddie/fred_code/mimic_llm_model.pth"
    best_recall = 0.0
    patience = 5
    stop_count = 0

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss, train_precision, train_recall = train_epoch(model, train_loader, optimiser, criterion)
        val_loss, val_precision, val_recall = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss) # update scheduler
    
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}')
        print(f'  Val Loss: {val_loss:.4f},   Val Precision: {val_precision:.4f},   Val Recall: {val_recall:.4f}\n')
    
        # save best model according to recall
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), model_path)
            stop_count = 0
        else:
            stop_count += 1

        if stop_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print(f'Training Complete. Best model (recall={best_recall:.4f}) saved to {model_path}\n')

    # final test
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_loss, test_precision, test_recall = evaluate(model, test_loader, criterion)

    print(f'Test Loss: {test_loss:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
