import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import D2NetDataset
from model_train import D2Net
from lib.loss import loss_function_2D
from tqdm import tqdm 
import matplotlib.pyplot as plt
import random
import os


# 変換を定義 (正規化やテンソルへの変換など)
transform = transforms.Compose([
   
   #transforms.RandomVerticalFlip(p=0.5),
   #transforms.RandomVerticalFlip(p=0.2),
   #transforms.RandomHorizontalFlip(0.3),

    transforms.ToTensor(),
    #transforms.RandomRotation(degrees=45),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = D2NetDataset(image_dir="", transform=transform,
       
)
print(len(train_dataset))
# データローダを定義 
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)


model = D2Net(use_cuda=torch.cuda.is_available())
#criterion = nn.MSELoss()  # ここではMSE損失を示していますが、適切な損失関数を選択する必要があります。
#optimizer = optim.Adam(model.parameters())
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


num_epochs =150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)
history={'loss':[]}

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
      

   
        loss = loss_function_2D(model, batch, device, plot=False)
       

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    history['loss'].append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

save_number=72


torch.save(model.state_dict(), 'd2net_model_own{}.pth'.format(save_number))

plt.plot(history['loss'],
         marker='.',
         label='loss(Training)')


plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("./loss_own{}.png".format(save_number))








