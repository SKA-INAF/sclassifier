import sys

# - IMPORT LUSTUFKA MODULES
sys.path.insert(1, '/home/riggi/Software/SKATools/mgcls_dino')
import utils
import vision_transformer as vits
#################################

import os
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx
        
        
######################################
###      DATASET
######################################
class AstroImageDataset(Dataset):
	""" Dataset to load astro images in FITS format """
	
	def __init__(self, filename, transform):
		self.filename= filename
		self.__read_filelist()
		self.transform = transform
		self.clip_data= False
		
	def __getitem__(self, idx):
		""" Override getitem method """
		
		# - Load PIL image at index
		image_pil= self.load_image(idx)
		
		# - Convert image for the model
		image_tensor= self.transform(image)
		
		# - Get label at inder idx
		class_id= self.datalist[idx]['id']
		
		return image_tensor, class_id
		
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
		
		# - Read FITS data
		data= fits.open(filename)[0].data
	
		# - Set NANs to image min
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]
		data_min= np.min(data_1d)
		data[~cond]= data_min
		
		# - Clip data?
		if self.clip_data:
			data_clipped= sigma_clipping(data_transf, sigma_low, sigma_up, sigma_bkg)
			data_transf= data_clipped
	
		# - Apply zscale stretch
		data_stretched= zscale_stretch(data_transf, contrast=contrast)
		data_transf= data_stretched 
	
		# - Convert to uint8
		data_transf= (data_transf*255.).astype(np.uint8)
	
		return data_transf
		
		
	def load_image(self, idx):
		""" Load image """
		
		# - Get image path
		item= self.datalist[idx]
		image_path= item["filepaths"][0]
		image_ext= os.path.splitext(image_path)[1]
		print("INFO: Reading image %s ..." % (image_path))
		
		# - Read FITS image as numpy array and then convert to PIL
		if image_ext=='.fits':
			data= self.__read_fits(image_path)
			image= Image.fromarray(data)
		else:
			image= Image.open(image_path)
			
		# - Convert to RGB image
		image_rgb= image.convert("RGB")
			
		return image_rgb
		
	def load_image_info(self, idx):
		""" Load image metadata """
		return self.datalist[idx]
		
	def __len__(self):
		return len(self.datalist)
			
	def get_sample_size(self):
		return len(self.datalist)


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features
    
    
    
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
	
	
	# - Model options
	parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='Per-GPU batch-size')
	parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
	parser.add_argument('--use_cuda', default=True, type=utils.bool_flag, help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
	parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
	parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
	parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    
	parser.add_argument('--dump_features', default=None, help='Path where to save computed features, empty for no saving')
	#parser.add_argument('--load_features', default=None, help="""If the features have already been computed, where to find them.""")
	parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
	parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
	parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
	
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
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Read args
	#filename_dataset= "/home/riggi/Data/MLData/radioimg-dataset/data/meerkat/G002/filelist_ann_multilabel_train.json"
	datalist= args.datalist
	
	# - Data options
	imgsize= args.imgsize
	nmax= args.nmax
	
	#===========================
	#==   BUILD MODEL
	#===========================
	print("INFO: Build network %s ..." % (args.arch))
	
	if "vit" in args.arch:
		model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
		print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
	elif "xcit" in args.arch:
		model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
	elif args.arch in torchvision_models.__dict__.keys():
		model = torchvision_models.__dict__[args.arch](num_classes=0)
		model.fc = nn.Identity()
	else:
		print(f"Architecture {args.arch} non supported")
		return 1
		
	model.cuda()
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
	
	transform = pth_transforms.Compose([
		pth_transforms.Resize(imgsize, interpolation=3),
		##pth_transforms.CenterCrop(224),
		pth_transforms.ToTensor(),
		pth_transforms.Normalize(data_mean, data_std),
	])
	
	dataset= AstroImageDataset(
		filename=datalist,
		transform=transform,
	)
	
	sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
	
	data_loader= torch.utils.data.DataLoader(
		dataset_train,
		##sampler=sampler,
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
	#train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)
	
	nsamples= len(dataset)
	
	for i in range(nsamples):
		if nmax!=-1 and i>=nmax:
			print("INFO: Max number of samples (%d) reached, exit loop..." % (nmax))
			
		imgs, class_ids = next(iter(data_loader))
		print("type(imgs)")
		print(type(imgs))
		print("type(class_ids)")
		print(type(class_ids))
		
		print("type(imgs[0])")
		print(type(imgs[0]))
		print(imgs[0].shape)
		
		print("INFO: Running inference ...")
		feats = model(imgs)
		
		#print(f"Feature batch shape: {imgs.size()}")
		#print(f"Labels batch shape: {class_ids.size()}")
		#img = imgs[0].squeeze()
		#class_id= class_ids[0]
	
	
	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
		
	
