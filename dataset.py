import os
import json

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

import torch
from torch.utils.data import Dataset

CLASSES = [
	'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
	'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
	'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
	'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
	'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
	'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
def classes():
	return CLASSES

def class_to_index(CLASSES):
	c2i = {v: i for i, v in enumerate(CLASSES)}
	return c2i

def index_to_class(c2i):
	i2c = {v: k for k, v in c2i.items()}
	return i2c 

def img_json_load(image_root, label_root):
	pngs = {
		os.path.relpath(os.path.join(root, fname), start=image_root)
		for root, _dirs, files in os.walk(image_root)
		for fname in files
		if os.path.splitext(fname)[1].lower() == ".png"
	} 

	jsons = {
		os.path.relpath(os.path.join(root, fname), start=label_root)
		for root, _dirs, files in os.walk(label_root)
		for fname in files
		if os.path.splitext(fname)[1].lower() == ".json"
	}

	# images/labels pair 확인
	jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
	pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

	assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
	assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

	pngs = sorted(pngs)
	jsons = sorted(jsons)
	return pngs, jsons
	
class BaseAugmentation(object):
	def __init__(self, img_size, is_train):
		self.is_train = is_train
		self.img_size = img_size
		self.transforms = self.get_transforms()

	def __call__(self, image, label=None):
		inputs = {"image": image}
		if label is not None:
			inputs["mask"] = label

		if self.transforms is not None:
			result = self.transforms(**inputs)
			image = result["image"]
			if label is not None:
				label = result["mask"]

		return image, label

	def get_transforms(self):
		if self.is_train:
			return A.Compose(
				[
					A.Resize(self.img_size, self.img_size),
				]
			)
		else:
			return A.Compose(
				[
					A.Resize(self.img_size, self.img_size),
				]
			)

class HardAugmentation(BaseAugmentation):
	def get_transforms(self):
		if self.is_train:
			return A.Compose(
				[
					A.Resize(self.img_size, self.img_size),
					# A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.7),
					A.RandomResizedCrop(1024, 1024, scale=(0.5, 1.0), ratio=(1.0, 1.0), always_apply=False, p=1.0),
					A.ElasticTransform(alpha=15.0, sigma=2.0),
					A.OneOf([A.Blur(blur_limit=2, p=1.0), A.MedianBlur(blur_limit=3, p=1.0)], p=0.2),
					A.HorizontalFlip(p=0.5),
				]
			)
		else:
			return A.Compose(
				[
					A.Resize(self.img_size, self.img_size),
				]
			)
 
class XRayDataset(Dataset):
	def __init__(self, is_train=True, transforms=None,
				 image_root='tarin/DCM',
				 label_root='train/outputs_json'): 

		pngs, jsons = img_json_load(image_root, label_root)
		_filenames = np.array(pngs)
		_labelnames = np.array(jsons)

		# split train-valid
		# 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
		# 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
		# 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
		groups = [os.path.dirname(fname) for fname in _filenames]

		# dummy label
		ys = [0 for __ in range(len(_filenames))]

		# 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
		# 5으로 설정하여 KFold를 수행합니다.
		gkf = GroupKFold(n_splits=5)
		
		filenames = []
		labelnames = []
		for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
			if is_train:
				# 0번을 validation dataset으로 사용합니다.
				if i == 0:
					continue
					
				filenames += list(_filenames[y])
				labelnames += list(_labelnames[y])
			
			else:
				filenames = list(_filenames[y])
				labelnames = list(_labelnames[y])
				
				# skip i > 0
				break
		
		self.image_root = image_root 
		self.label_root = label_root
		self.filenames = filenames
		self.labelnames = labelnames
		self.is_train = is_train
		self.transforms = transforms
	
	def __len__(self):
		return len(self.filenames)
	
	def __getitem__(self, item):
		image_name = self.filenames[item]
		image_path = os.path.join(self.image_root, image_name)
		
		image = cv2.imread(image_path)
		image = image / 255.
		
		label_name = self.labelnames[item]
		label_path = os.path.join(self.label_root, label_name)
		
		# (H, W, NC) 모양의 label을 생성합니다.
		label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
		label = np.zeros(label_shape, dtype=np.uint8)
		
		# label 파일을 읽습니다.
		with open(label_path, "r") as f:
			annotations = json.load(f)
		annotations = annotations["annotations"]
		
		# 클래스 별로 처리합니다.
		class2ind = class_to_index(CLASSES)
		for ann in annotations:
			c = ann["label"]
			class_ind = class2ind[c]
			points = np.array(ann["points"])
			
			# polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
			class_label = np.zeros(image.shape[:2], dtype=np.uint8)
			cv2.fillPoly(class_label, [points], 1)
			label[..., class_ind] = class_label

		if self.transforms is not None:
			inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
			result = self.transforms(**inputs)
			
			image = result["image"]
			label = result["mask"] if self.is_train else label

		# to tenser will be done later
		image = image.transpose(2, 0, 1)	# channel first 포맷으로 변경합니다.
		label = label.transpose(2, 0, 1)
		
		image = torch.from_numpy(image).float()
		label = torch.from_numpy(label).float()
			
		return image, label

class XRayInferenceDataset(Dataset):
	def __init__(self, transforms=None, image_root='test/DCM', label_root='train/outputs_json'):
		pngs, __ = img_json_load(image_root, label_root)
		_filenames = pngs
		_filenames = np.array(sorted(_filenames))
		
		self.image_root = image_root
		self.filenames = _filenames
		self.transforms = transforms
	
	def __len__(self):
		return len(self.filenames)
	
	def __getitem__(self, item):
		image_name = self.filenames[item]
		image_path = os.path.join(self.image_root, image_name)
		
		image = cv2.imread(image_path)
		image = image / 255.
		
		if self.transforms is not None:
			inputs = {"image": image}
			result = self.transforms(**inputs)
			image = result["image"]

		# to tenser will be done later
		image = image.transpose(2, 0, 1)  
		
		image = torch.from_numpy(image).float()
			
		return image, image_name