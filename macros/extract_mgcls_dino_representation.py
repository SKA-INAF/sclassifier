import sys

# - IMPORT LUSTUFKA MODULES
sys.path.insert(1, '/home/riggi/Software/Sources/mgcls_dino')
import utils
import vision_transformer as vits
#################################

import os
import argparse
import json
import warnings
import numpy as np


## ASTRO ####
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval

## IMAGE PROC ###
from PIL import Image

## TORCH ####
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torch.utils.data import Dataset, DataLoader


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx
        
        
######################################
###      DATASET
######################################
class AstroImageDataset(Dataset):
	""" Dataset to load astro images in FITS format """
	
	def __init__(self, filename, transform, in_chans=1, apply_zscale=False, norm_range=(0.,1.), to_uint8=False):
		self.filename= filename
		self.__read_filelist()
		self.transform = transform
		self.clip_data= False
		self.in_chans= in_chans
		self.apply_zscale= apply_zscale
		self.norm_range= norm_range
		self.to_uint8= to_uint8

	def __getitem__(self, idx):
		""" Override getitem method """

		# - Get label at inder idx
		class_id= self.datalist[idx]['id']

		# - Get object identifier
		sname= self.datalist[idx]['sname']

		# - Load PIL image at index
		image_pil, is_good_data= self.load_image(idx)
		if image_pil is None:
			print("WARN: Failed to load image ...")
			is_good_data= False

		# - Convert image for the model
		image_tensor= self.transform(image_pil)

		return image_tensor, class_id, sname, is_good_data

	def __read_filelist(self):
		""" Read input json filelist """
		fp= open(self.filename, "r")
		self.datalist= json.load(fp)["data"]	

	def __get_clipped_data(self, data, sigma_low=5, sigma_up=30):
		""" Apply sigma clipping to input data and return transformed data """

		# - Find NaNs pixels
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]

		# - Clip all pixels that are below sigma clip
		res= sigma_clip(data_1d, sigma_lower=sigma_low, sigma_upper=sigma_up, masked=True, return_bounds=True)
		thr_low= res[1]
		thr_up= res[2]

		data_clipped= np.copy(data)
		data_clipped[data_clipped<thr_low]= thr_low
		data_clipped[data_clipped>thr_up]= thr_up

		# - Set NaNs to 0
		data_clipped[~cond]= 0

		return data_clipped

	def __get_zscaled_data(self, data, contrast=0.25):
		""" Apply sigma clipping to input data and return transformed data """

		# - Find NaNs pixels
		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Apply zscale transform
		transform= ZScaleInterval(contrast=contrast)
		data_transf= transform(data)	

		# - Set NaNs to 0
		data_transf[~cond]= 0

		return data_transf

	def __read_fits(self, filename):
		""" Read FITS image """

		is_good_data= True

		# - Read FITS data
		data= fits.open(filename)[0].data

		# - Set NANs to image min
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]
		if data_1d.size==0:
			is_good_data= False
			print("WARN: All NAN image, setting image to 0...")
			data[~cond]= 0
			return data.astype(np.uint8), is_good_data
			#return None

		data_min= np.min(data_1d)
		data[~cond]= data_min

		data_transf= data
		
		print("== DATA MIN/MAX ==")
		print(data_transf.min())
		print(data_transf.max())

		# - Clip data?
		if self.clip_data:
			data_clipped= self.__get_clipped_data(data_transf, sigma_low=5, sigma_up=30)
			data_transf= data_clipped

		# - Apply zscale stretch
		if self.apply_zscale:
			data_stretched= self.__get_zscaled_data(data_transf, contrast=0.25)
			data_transf= data_stretched

		# - Convert to uint8
		#data_transf= (data_transf*255.).astype(np.uint8)
		
		# - Normalize to range
		data_min= data_transf.min()
		data_max= data_transf.max()
		norm_min= self.norm_range[0]
		norm_max= self.norm_range[1]
		if norm_min==data_min and norm_max==data_max:
			print("INFO: Data already normalized in range (%f,%f)" % (norm_min, norm_max))
		else:
			data_norm= (data_transf-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
			data_transf= data_norm
			
		print("== DATA MIN/MAX (AFTER TRANSF) ==")
		print(data_transf.min())
		print(data_transf.max())
	
		# - Convert to uint8
		if self.to_uint8:
			data_transf= data_transf.astype(np.uint8)
	
		return data_transf, is_good_data


	def load_image(self, idx):
		""" Load image """

		# - Get image path
		item= self.datalist[idx]
		image_path= item["filepaths"][0]
		image_ext= os.path.splitext(image_path)[1]
		print("INFO: Reading image %s ..." % (image_path))

		# - Read FITS image as numpy array and then convert to PIL
		is_good_data= True
		if image_ext=='.fits':
			data, is_good_data= self.__read_fits(image_path)
			if data is None or not is_good_data:
				print("WARN: Failed to read FITS data ...")
				#return None
			image= Image.fromarray(data)
		else:
			image= Image.open(image_path)

		# - Convert to RGB image
		if self.in_chans==3:
			image= image.convert("RGB")

		print("--> image.shape")
		print(np.asarray(image).shape)	

		return image, is_good_data

	def load_image_info(self, idx):
		""" Load image metadata """
		return self.datalist[idx]
		
	def __len__(self):
		return len(self.datalist)
			
	def get_sample_size(self):
		return len(self.datalist)

def write_ascii(data, filename, header=''):
	""" Write data to ascii file """

	# - Skip if data is empty
	if data.size<=0:
		print("WARN: Empty data given, no file will be written!")
		return

	# - Open file and write header
	fout = open(filename, 'wt')
	if header:
		fout.write(header)
		fout.write('\n')	
		fout.flush()	
		
	# - Write data to file
	nrows= data.shape[0]
	ncols= data.shape[1]
	for i in range(nrows):
		fields= '  '.join(map(str, data[i,:]))
		fout.write(fields)
		fout.write('\n')	
		fout.flush()	

	fout.close()


###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-datalist','--datalist', dest='datalist', required=True, type=str, help='Input data json filelist') 
	parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
	
	# - Data options
	parser.add_argument('--imgsize', default=224, type=int, help='Image resize size in pixels')
	parser.add_argument('--nmax', default=-1, type=int, help='Number of images to read and process in input file (-1=all)')
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--norm_min', default=0., type=float, help='Norm min (default=0)')
	parser.add_argument('--norm_max', default=1., type=float, help='Norm max (default=1)')
	parser.add_argument('--to_uint8', dest='to_uint8', action='store_true',help='Convert to uint8 (default=false)')	
	parser.set_defaults(to_uint8=False)
	parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
  
	# - Model options
	parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='Per-GPU batch-size')
	parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
	parser.add_argument('--use_cuda', default=True, type=utils.bool_flag, help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
	parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
	parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
	parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
	
	parser.add_argument('--dump_features', default=None, help='Path where to save computed features, empty for no saving')
	parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
	
	# - Outfile option
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata.dat', help='Output filename (.dat) of feature data') 


	args = parser.parse_args()	

	return args
	
	
##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	print("INFO: Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Read args
	datalist= args.datalist
	
	# - Data options
	imgsize= args.imgsize
	nmax= args.nmax
	
	#===========================
	#==   BUILD MODEL
	#===========================
	print("INFO: Build network %s ..." % (args.arch))
	
	if "vit" in args.arch:
		model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans=args.in_chans)
		print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
	elif "xcit" in args.arch:
		model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
	elif args.arch in torchvision_models.__dict__.keys():
		model = torchvision_models.__dict__[args.arch](num_classes=0)

		if args.in_chans != 3:
			model.conv1 = nn.Conv2d(args.in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		if args.arch == "resnet18": #after converting the checkpoint keys to Torchvision names
			model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

		model.fc = nn.Identity()
	else:
		print(f"Architecture {args.arch} non supported")
		return 1

	if args.use_cuda:
		model.cuda()


	print("model")
	print(model)

	print("INFO: Load pretrained weights from file %s ..." % (args.pretrained_weights))
	utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
	model.eval()

	#===========================
	#==   SET DATA LOADER
	#===========================
	data_mean= (0.485, 0.456, 0.406)
	data_std= (0.229, 0.224, 0.225)
	#data_mean= (0.0, 0.0, 0.0)
	#data_std= (1.0, 1.0, 1.0) 
	
	transform_RGB = pth_transforms.Compose([
		pth_transforms.Resize(imgsize, interpolation=3),
		pth_transforms.ToTensor(),
		#pth_transforms.Normalize(data_mean, data_std),
	])
	
	transform_gray = pth_transforms.Compose([
		pth_transforms.Resize(imgsize, interpolation=3),
		pth_transforms.ToTensor(),
		#pth_transforms.Normalize(data_mean, data_std),
	])
	
	if args.in_chans==1:
		transform= transform_gray
	elif args.in_chans==3:
		transform= transform_RGB
	else:
		print("ERROR: Invalid/unknown in_chan (%d)!" % (args.in_chans))
	
	dataset= AstroImageDataset(
		filename=datalist,
		transform=transform,
		in_chans=args.in_chans,
		apply_zscale=args.zscale,
		norm_range=(args.norm_min, args.norm_max),
		to_uint8=args.to_uint8
	)
	
	data_loader= torch.utils.data.DataLoader(
		dataset,
		shuffle=False,
		batch_size=args.batch_size_per_gpu,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)
    
	print(f"Data loaded with {len(dataset)} imgs.")
	
	#===========================
	#==   EXTRACT FEATURES
	#===========================
	nsamples= len(dataset)
	feature_list= []
	sname_list= []
	classid_list= []

	iterator= iter(data_loader)

	for i in range(nsamples):
		# - Stop looping?
		if nmax!=-1 and i>=nmax:
			print("INFO: Max number of samples (%d) reached, exit loop..." % (nmax))
			break

		# - Load data from loader
		ret= next(iterator)
		if ret is None:
			print("Failed to read image %d, skipping ..." % (i+1))
			continue
		
		imgs= ret[0]
		class_ids= ret[1]
		sname = ret[2]	
		is_good_data= ret[3]
		if not is_good_data:
			print("Bad data image %d, skipping ..." % (i+1))
			continue

		
		# - Run inference
		with torch.no_grad():
			feats = model(imgs)

		features_numpy= feats[0].cpu().numpy()
		class_ids_numpy= class_ids[0].cpu().numpy()

		if i==0:
			print("feats.shape")
			print(feats.shape)
			print("features_numpy.shape")
			print(features_numpy.shape)

		# - Append to main list
		#feature_list.append(features_numpy)
		#sname_list.append(sname)
		#classid_list.append(class_ids_numpy)
		feature_list.extend(features_numpy)
		sname_list.extend(sname)
		classid_list.extend(class_ids_numpy)

	#===========================
	#==   SAVE FEATURES
	#===========================
	# - Write selected feature data table
	print("INFO: Writin feature data to file %s ..." % (args.outfile))

	N= len(feature_list)
	nfeats= feature_list[0].shape[0]
	print("INFO: N=%d, nfeats=%d" % (N, nfeats))

	featdata_arr= np.array(feature_list)
	snames_arr= np.array(sname_list).reshape(N,1)
	classids_arr= np.array(classid_list).reshape(N,1)

	outdata= np.concatenate(
		(snames_arr, featdata_arr, classids_arr),
		axis=1
	)

	znames_counter= list(range(1,nfeats+1))
	znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
	head= '{} {} {}'.format("# sname",znames,"id")

	write_ascii(outdata, args.outfile, head)


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())


