# python native
import os
from argparse import ArgumentParser
import os.path as osp

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# torch
import torch
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# Custom file
import dataset
from dataset import XRayInferenceDataset, BaseAugmentation

CHECKPOINT_EXTENSIONS = ['.pth', '.pt']

CLASSES = dataset.classes()

def parse_args():
	parser = ArgumentParser()

	# Conventional 	
	parser.add_argument('--data_dir', type=str, default='test/DCM')
	parser.add_argument('--model_dir', type=str, default='trained_model/fcn_resnet50_best_model.pt')
	parser.add_argument('--output_dir', type=str, default='outputs')

	parser.add_argument('--device', default=cuda.is_available())
	parser.add_argument('--img_size', type=int, default=2048)
	parser.add_argument('--batch_size', type=int, default=5)

	args = parser.parse_args()

	return args

def encode_mask_to_rle(mask):
	'''
	mask: numpy array binary mask 
	1 - mask 
	0 - background
	Returns encoded run length 
	'''
	pixels = mask.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
	s = rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(height * width, dtype=np.uint8)
	
	for lo, hi in zip(starts, ends):
		img[lo:hi] = 1
	
	return img.reshape(height, width)

def do_inference(model, data_loader, thr=0.5, device='cpu'):
	model = model.to(device)
	model.eval()

	class2ind = dataset.class_to_index(CLASSES)
	ind2class = dataset.index_to_class(class2ind)

	rles = []
	filename_and_class = []

	with torch.no_grad():
		for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
			images = images.to(device)    
			outputs = model(images)['out']
			
			outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
			outputs = torch.sigmoid(outputs)
			outputs = (outputs > thr).detach().cpu().numpy()
			
			for output, image_name in zip(outputs, image_names):
				for c, segm in enumerate(output):
					rle = encode_mask_to_rle(segm)
					rles.append(rle)
					filename_and_class.append(f"{ind2class[c]}_{image_name}")
					
	return rles, filename_and_class

def main(data_dir, model_dir, output_dir, device, img_size, batch_size):
	
	# 테스트 데이터 경로를 입력하세요

	devickjje = torch.device('cuda:0' if device else 'cpu')
	image_root = data_dir
	
	if not osp.exists(output_dir):
		os.makedirs(output_dir)
	
	# Initialize model
	model = torch.load(model_dir)

	tf = BaseAugmentation(img_size=img_size, is_train=False).transforms

	test_dataset = XRayInferenceDataset(
		transforms=tf,
		image_root=image_root
	)

	test_loader = DataLoader(
		dataset=test_dataset, 
		batch_size=batch_size,
		shuffle=False,
		num_workers=2,
		drop_last=False
	)
	rles, filename_and_class = do_inference(model, test_loader, device)
	
	classes, filename = zip(*[x.split("_") for x in filename_and_class])
	image_name = [os.path.basename(f) for f in filename]
	df = pd.DataFrame({
		"image_name": image_name,
		"class": classes,
		"rle": rles,
	})
	output_fname = "output.csv"
	df.to_csv(osp.join(output_dir, output_fname), index=False)
	

if __name__ == '__main__':
	args = parse_args()
	main(**args.__dict__)