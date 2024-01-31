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
   
    #transforms.ToTensor(),
   #transforms.RandomVerticalFlip(p=0.5),
   #transforms.RandomVerticalFlip(p=0.2),
   #transforms.RandomAffine(
    #degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)
   #),
   #transforms.RandomHorizontalFlip(0.3),

    transforms.ToTensor(),
    #transforms.RandomRotation(degrees=45),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



#image_path1="/mnt/c/Users/kamanberu88/Desktop/jpg8k_me/400"
#image_path1="/mnt/c/Users/kamanberu88/Desktop/croppd_images"
#image_path2="/mnt/c/Users/kamanberu88/Desktop/JAXA_database/mapimg/CST1/TCO_CST1_TM_SIM_a7351_i3438_H36.bmp"
#train_dataset=D2NetDataset(image_path1,image_path2,transform=transform)
#image_path1="/mnt/c/Users/kamanberu88/Desktop/jpg8k_me/406"
#image_path2="/mnt/c/Users/kamanberu88/Desktop/jpg8k_me/400"
#train_dataset=D2NetDataset(image_path1,image_path2,transform=transform)
# データセットを定義
#train_dataset = D2NetDataset(image_dir="/home/kamanberu88/new_dataset/400/", transform=transform)
       #save_samples=True,sample_save_dir='./add_noise_gazo/'
#)
#/home/kamanberu88/new_dataset/400/
train_dataset = D2NetDataset(image_dir="/home/kamanberu88/new_dataset/400/", transform=transform,
       
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
        #outputs = model(batch)

        # この例では、損失関数としてMSEを使用していますが、実際のタスクや目的に応じて適切な損失関数を定義する必要があります。
        # 例えば、一致点の一致度を計算するためのカスタム損失関数を使用する場合が考えられます。
        #loss = criterion(outputs['dense_features1'], outputs['dense_features2'])
        loss = loss_function_2D(model, batch, device, plot=False)
        #loss=loss_function_2d(model,batch,device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    history['loss'].append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    save_number=72
    #if epoch==51:

     #   torch.save(model.state_dict(), 'd2net_model_own{}_ep{}_.pth'.format(save_number,epoch))
    #if epoch==101:

     #   torch.save(model.state_dict(), 'd2net_model_own{}_ep{}_.pth'.format(save_number,epoch))

    #elif epoch==151:
      #  torch.save(model.state_dict(), 'd2net_model_own{}_ep{}_.pth'.format(save_number,epoch))
    




torch.save(model.state_dict(), 'd2net_model_own{}.pth'.format(save_number))

plt.plot(history['loss'],
         marker='.',
         label='loss(Training)')


plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("./loss_own{}.png".format(save_number))








