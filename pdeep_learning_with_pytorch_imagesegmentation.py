"""PDeep Learning with PyTorch-ImageSegmentation.ipynb
# Task 1 : Set up colab gpu runtime environment

!pip install segmentation-models-pytorch
!pip install -U git+https://github.com/albumentations-team/albumentations
!pip install --upgrade opencv-contrib-python

"""# Download Dataset
#train, csv files comma separated values
!git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git
# Some Common Imports
"""

import sys
sys.path.append('/content/Human-Segmentation-Dataset-master')
#sys managing file path, command line arg like LR btch size,, runtime in pythion

import torch #tenosr,face module.nn, backward(),
import cv2 #face detection, edge, image processing

import numpy as np
import pandas as pd #structured data, matplotlib, feature engg, read CSV file df.
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #1hot encode .transform()
from tqdm import tqdm #loops n iterative tasks

import helper #code reusability, here folder for visulaizatn fn

"""# Task : 2 Setup Configurations"""

CSV_FILE = '/content/Human-Segmentation-Dataset-master/train.csv'
DATA_DIR = '/content/'
DEVICE = 'cuda' #GPU
EPOCHS = 25
LR = 0.003
IMG_SIZE = 320 #H AND W
BATCH_SIZE = 16
ENCODER = 'timm-efficientnet-b0' #enc transform raw data into a lower-dimensional, more informative representation that can be easily processed by other parts of a model
WEIGHTS = 'imagenet' #contains millions of images of thsnds of classs, pre-trained weights of a neural network that has been trained on the ImageNet dataset

#PLOT IMAGES
df = pd.read_csv(CSV_FILE) #panda comma separated value
df.head() #gives first few egs, will get img n mask PATHS

#Understand the dataset:
row = df.iloc[7] #pandas dataframe of 7th row eg.
image_path = row.images #takes path frm above op of coloumn named imags
mask_path = row.masks
#to read image use cv2
image = cv2.imread(image_path) #to load image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #opencv reads in BGR
#to read mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0  #we want it in grey scale n normalised

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.set_title('IMAGE')
ax1.imshow(image)

ax2.set_title('GROUND TRUTH')
ax2.imshow(mask,cmap = 'gray')

#split dara to train an val
train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42) #TTS SCIKIT, 20% test set

"""# Task 3 : Augmentation Functions

albumentation documentation : https://albumentations.ai/docs/
"""

#aug to increase train set by artifically modifying. prevents overfittng, enhance generalztn#
#for claasfctn tasks augmntn on images never on labels ; also on labels for on segmentatnn/ localizatn tasks
import albumentations as A #augmentatn libarary for cv tasks

def get_train_augs():
  return A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE), #alraedy spcfd in confgtn
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.5)

  ])
def get_valid_augs():  #in valid set no augmntn is done
  return A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE),
#used in inference so only resize

  ])

"""# Task 4 : Create Custom Dataset"""

#done to get image mask pair acc to index as done above actually, here using class method
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
  #3methods:
  def __init__(self, df, augmentations): #init first is self
    self.df = df #storing the dataframe as an instance variable
    self.augmentations = augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx): #helps to get indexed  as key, label;as done above to get ondex image n mask
    row = self.df.iloc[idx] #selecting the row

    image_path = row.images
    image2 = cv2.imread(image_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) #hwc

    mask_path = row.masks
    mask2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #in hw
    mask2 = np.expand_dims(mask, axis = -1) #now hwc, added one more dim

    if self.augmentations: #if augmnts r true, gonna apply aug on data n mask
      data = self.augmentations(image = image2, mask = mask2) #returns dict image, mask as key, vvalue
  #extract img n mask accrdng to key
      image = data['image'] #hwc==0,1,2 to 201 or chw in transpose below
      mask = data['mask']

      ##pytoch uses chw
    image = np.transpose(image, (2,0,1)).astype(np.float32) #chw
    mask = np.transpose(mask, (2,0,1)).astype(np.float32) #chw

      #to convert this numpy to tensor and scale it
    image = torch.Tensor(image) / 255.0 #to normalise get renge 0 t0 1
    mask = torch.round(torch.Tensor(mask) / 255.0) #also rounding up values for mask
    return image, mask

#use above class to get train n valid dataset
trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())

#to get no.of egs in train n valid sets
print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")

#plot few images
idx = 3

image, mask = trainset[idx]
helper.show_image(image, mask)

"""# Task 5 : Load dataset into batches"""

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size = BATCH_SIZE)

print(f"Total number of batches in trainloader : {len(trainloader)}")
print(f"Total number of batches in validloader : {len(validloader)}")
#15= 232 above/ 16 in confgtn

for image, mask in trainloader:
  break #coz we want only one
print(f"one batch image shape : {image.shape}") #torch.size16,3,320,320
print(f"one batch mask shape : {mask.shape}")  #torch.size16,1,320,320
#Mostly for segmentation problem, input image size and mask image size should be same.

"""# Task 6 : Create Segmentation Model

#using efficientnet as encoder and segmentatn model as Unet, sgmntn loss selected is Diceloss
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(nn.Module):
  def __init__(self):
    super(SegmentationModel, self).__init__() #basic stuff while creatng any model in pytorch

    self.arc = smp.Unet( #architectre
         encoder_name = ENCODER, #defnd in confgtn
         encoder_weights = WEIGHTS,
         in_channels = 3, #color input RGB
         classes = 1, #BINARY SEGMNTN SO CLASSES 1
         activation = None #final a layer so op will b logits
     )
    def forward(self, images, masks = None):
      logits = self.arc(images) #op will b logits wen passd thru this arc

      if masks != None:
        loss1 = DiceLoss(mode= 'binary')(logits, masks) #binary problm here, loss calculatd btwn logits and true masks
        loss2 = nn.BCEwithLogitsLoss()(logits, masks) #combined both loss fntns
        return logits, loss1 + loss2

      return logits #ifmask is none, true during testng/ infrnce

model = SegmentationModel()
#model.to(DEVICE) #move to cuda

"""# Task 7 : Create Train and Validation Function"""

def train_fn(data_loader, model, optimizer):
  model.train() #indicates dropout layer is on, model should b in training mode
  total_loss = 0.0

  for images, mask in tqdm(data_loader): #tqdm to trach batchs
    images = images.to(DEVICE)
    mask = mask.to(DEVICE)

    optimizer.zero.grad()
    logits, loss = model(images, mask) #L,L WILL B RETRND WEN I,M IP mask is gtruth, logits is pred truth
    loss.backward() #TO FIND GRADIENTS,#got loss , find gradients
    optimizer.step() #to update weights n biases parametrs of d model
    total_loss += loss.item() #every batch loss will b here
  return total_loss / len(data_loader) #avg loss retrnd = tloss/ no. of batches

def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad(): #no gradients here
      for images, mask in tqdm(data_loader): #tqdm to trach batchs
        images = images.to(DEVICE)
        mask = mask.to(DEVICE)

        logits, loss = model(images, mask) #L,L WILL B RETRND WEN I,M IP

        total_loss += loss.item()
    return total_loss / len(data_loader)

"""# Task 8 : Train Model"""

optimizer = torch.optim.Adam(model.parameters(), lr = LR) #parametrs r the w and b

best_valid_loss = np.Inf
for i in range(EPOCHS):
  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss = eval_fn(validloader, model)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt') #in left folder #named it as best model, will b used in infrnce
    print("SAVED MODEL") #weights are saved
    best_valid_loss = valid_loss

    print(f"Epoch : {i+1} Train Loss : {train_loss} Valid Loss : {valid_loss}")
#op first progrssn bar is train fn, second is eval fn


"""# Task 9 : Inference"""
#to compare acual mask with predicted  mask
#above got best model at epoch = 19, using that here
idx = 20
model.load_state_dict(torch.load('/content/best_model.pt')) #to load best weights
image, mask = validset[idx] #here img in chw #image tensor from train/valid has shape chw
logits_mask = model(image.to(DEVICE).unsqueeze(0)) #here img tensor must batchsize,chw 
#unsqueeze at xis 0 adding 1 batch dimensn to 1, c,h,w
#above logits given sigmoid:
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5)*1.0 #putting a threshold , above whovh is considered as one

helper.show_image(image, mask, pred_mask.detach().cpu().squeeze(0)) #original mask, pred mask in bchw,,,to remove batch b used unsqueeze
#shows color people image, bnw ground truth image which is mask, model op which is pred mask. last two similar

