# python native
import os
import json
import time
import random
import math
import datetime
from argparse import ArgumentParser

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from torchvision import models
from torchvision.models.segmentation import FCN_ResNet50_Weights 

# etc
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

# Custom files
import dataset
from dataset import XRayDataset, HardAugmentation, BaseAugmentation
import visualization
import model as m

RANDOM_SEED = 2024
CLASSES = dataset.classes()


def parse_args():
	parser = ArgumentParser()

	# Conventional args
	parser.add_argument('--data_root', type=str, default='train/DCM')
	parser.add_argument('--label_root', type=str, default='train/outputs_json')
	parser.add_argument('--save_dir', type=str, default='trained_model')
	# parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--learning_rate', type=float, default=1e-4)
	parser.add_argument('--num_epochs', type=int, default=20)
	parser.add_argument('--val_every', type=int, default=5)

	parser.add_argument('--device', default=cuda.is_available())
	parser.add_argument('--save_wandb_name', type=str, default='ss')
	parser.add_argument('--model_name', type=str, default='fcn_resnet50')

	args = parser.parse_args()

	return args
	
def get_preprocessing(preprocessing_fn):
	"""Construct preprocessing transform
	
	Args:
		preprocessing_fn (callbale): data normalization function 
			(can be specific for each pretrained neural network)
	Return:
		transform: albumentations.Compose
	
	"""
	
	_transform = [
		A.Lambda(image=preprocessing_fn),
		#A.Lambda(image=to_tensor, mask=to_tensor),
	]
	return A.Compose(_transform)

def to_tensor(x, **kwargs):
	return x.transpose(2, 0, 1).astype('float32')

def save_model(model, file_name='fcn_resnet50_best_model.pt', save_dir='trained_model'):
	output_path = os.path.join(save_dir, file_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	torch.save(model, output_path)

def set_seed():
	torch.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(RANDOM_SEED)
	random.seed(RANDOM_SEED)

def dice_coef(y_true, y_pred):
	y_true_f = y_true.flatten(2)
	y_pred_f = y_pred.flatten(2)
	intersection = torch.sum(y_true_f * y_pred_f, -1)

	eps = 0.0001
	return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def transforms_validation(data_root, label_root, tf):

	set_seed()

	train_dataset = XRayDataset(is_train=True, 
							 transforms=tf,
							 image_root=data_root,
							 label_root=label_root)

	image, label = train_dataset[0]
	fig, ax = plt.subplots(1, 2, figsize=(24, 12))
	ax[0].imshow(image[0])    # color map 적용을 위해 channel 차원을 생략합니다.
	ax[1].imshow(visualization.label2rgb(label))

	plt.show()

def validation(epoch, model, data_loader, criterion, thr=0.5, device='cpu'):
	print(f'Start validation #{epoch:2d}')
	model.eval()

	dices = []
	with torch.no_grad():
		total_loss = 0
		cnt = 0

		for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
			images, masks = images.to(device), masks.to(device)
			model = model.to(device)

			outputs = model(images)['out']
			
			output_h, output_w = outputs.size(-2), outputs.size(-1)
			mask_h, mask_w = masks.size(-2), masks.size(-1)
			
			# gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
			if output_h != mask_h or output_w != mask_w:
				outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
			
			loss = criterion(outputs, masks)
			total_loss += loss
			cnt += 1

			outputs = torch.sigmoid(outputs)
			outputs = (outputs > thr).detach().cpu()
			masks = masks.detach().cpu()

			dice = dice_coef(outputs, masks)
			dices.append(dice)

	dices = torch.cat(dices, 0)
	dices_per_class = torch.mean(dices, 0)
	dice_str = [
		f"{c:<12}: {d.item():.4f}"
		for c, d in zip(CLASSES, dices_per_class)
	]
	dice_str = "\n".join(dice_str)
	print(dice_str)

	avg_dice = torch.mean(dices_per_class).item()
	return avg_dice

def do_training(data_root, label_root, save_dir, device, batch_size, 
				learning_rate, num_epochs, val_every, save_wandb_name, model_name):

	set_seed()

	tf_train = HardAugmentation(img_size=1024, is_train=True).transforms
	tf_valid = HardAugmentation(img_size=1024, is_train=False).transforms

	train_dataset = XRayDataset(is_train=True, 
							 transforms=tf_train,
							 image_root=data_root,
							 label_root=label_root)
	valid_dataset = XRayDataset(is_train=False,
							 transforms=tf_valid,
							 image_root=data_root,
							 label_root=label_root)

	train_loader = DataLoader(
		dataset=train_dataset, 
		batch_size=batch_size,
		shuffle=True,
		num_workers=8,
		drop_last=True,
	)

	# 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
	valid_loader = DataLoader(
		dataset=valid_dataset, 
		batch_size=6,
		shuffle=False,
		num_workers=0,
		drop_last=False
	)

	wandb.init(project='segmentation')
	wandb.run.name = save_wandb_name
	wandb.config.update({
			"learning_rate" : learning_rate,
			"eochs" : num_epochs,
			"batch_size" : batch_size	
	})

	device = torch.device('cuda:0' if device else 'cpu')
	n_class = len(CLASSES)
	model = m.create_model(model_name)
	model = model.to(device)

	# Loss function을 정의합니다.
	criterion = nn.BCEWithLogitsLoss()

	# Optimizer를 정의합니다.
	optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)

	# training
	print(f'Start training..') 

	best_dice = 0.
	
	for epoch in range(num_epochs):
		model.train()

		train_dict = {
			'train_loss' : 0,
			'evel_dice' : 0,
		}

		with tqdm(total=len(train_loader)) as pbar:
			for step, (images, masks) in enumerate(train_loader):
				# gpu 연산을 위해 device 할당합니다.
				images, masks = images.to(device), masks.to(device)
				model = model.to(device)

				# torchvision의 모델을 사용하는 경우, 'out' key로 해서 불러 와야 함. ex) outputs = model(images)['out']
				outputs = model(images)
				
				# loss를 계산합니다.
				loss = criterion(outputs, masks)

				pbar.set_description(
						f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
						f'Epoch [{epoch+1}/{num_epochs}], '
						f'Step [{step+1}/{len(train_loader)}], '
						f'Loss: {round(loss.item(),4)}')

				pbar.update(1)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			# validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
			if (epoch + 1) % val_every == 0:
				dice = validation(epoch + 1, model, valid_loader, criterion, device=device)

				if best_dice < dice:
					print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
					print(f"Save model in {save_dir}")
					best_dice = dice
					save_model(model)


def main(args):
	do_training(**args.__dict__)

if __name__ == '__main__':
	args = parse_args()
	main(args)