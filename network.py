
import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import faiss
from math import ceil
from tqdm import tqdm
from netvlad import NetVLAD
from pooling import GeM
from CBAM import CBAMBlock 

class GeoLocalizationNet(nn.Module):
    """The model is composed of a backbone and an aggregation layer.
    The backbone is a (cropped) ResNet-18, and the aggregation is a L2
    normalization followed by max pooling.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.attention = None

        if args.attention:
            print("Using CBAM attention module")
            self.attention = CBAMBlock(channel=256)
            self.attention.init_weights()

        if args.mode == "netvlad":
            logging.debug(f"Using NetVLAD aggregation with {args.num_clusters} clusters")
            netvlad = NetVLAD(num_clusters=args.num_clusters, dim=args.features_dim)
            if args.resume is None:
                logging.debug("Clustering for NetVLAD initialization")
                centroids, descriptors = get_clusters(args, self.backbone)
                netvlad.init_params(centroids, descriptors)
                del args.cluster_ds
            self.aggregation = nn.Sequential(L2Norm(), netvlad)
            args.features_dim *= args.num_clusters

        elif args.mode == "gem":
            logging.debug("Using GeM aggregation")
            self.aggregation = nn.Sequential(L2Norm(), GeM(), Flatten())

        elif args.mode == "avg_pool":
            logging.debug("Using Avg Pooling aggregation")
            self.aggregation = nn.Sequential(L2Norm(),
                                            nn.AdaptiveAvgPool2d(1),
                                            Flatten())

        else:
            raise RuntimeError(f"Unknown mode {args.mode}")

    def forward(self, x):
        x = self.backbone(x)
        if self.attention:
            x = self.attention(x)
        x = self.aggregation(x)
        return x


def get_backbone(args):
    backbone = torchvision.models.resnet18(pretrained=True)
    for name, child in backbone.named_children():
        if name == "layer3":
            break
        for params in child.parameters():
            params.requires_grad = False
    logging.debug("Train only conv4 of the ResNet-18 (remove conv5), freeze the previous ones")
    layers = list(backbone.children())[:-3]
    backbone = nn.Sequential(*layers)
    args.features_dim = 256  # Number of channels in conv4
    return backbone


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:,:,0,0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def get_clusters(args, model):
    num_descriptors = 50000
    desc_per_image = 100
    num_images = ceil(num_descriptors / desc_per_image)
    
    sampler = SubsetRandomSampler(np.random.choice(len(args.cluster_ds), num_images, replace=False))
    data_loader = DataLoader(dataset=args.cluster_ds, num_workers=args.num_workers,
                            batch_size=args.infer_batch_size, shuffle=False,
                            sampler=sampler, pin_memory=(args.device=="cuda"))
    with torch.no_grad():
        model = model.eval().to(args.device)
        descriptors = np.zeros(shape=(num_descriptors, args.features_dim), dtype=np.float32)
        for iteration, (inputs, indices) in enumerate(tqdm(data_loader, ncols=100), 1):
            inputs = inputs.to(args.device)
            encoder_out = model(inputs)
            l2_out = F.normalize(encoder_out, p=2, dim=1)
            image_descriptors = l2_out.view(l2_out.size(0), args.features_dim, -1).permute(0, 2, 1)
            batchix = (iteration - 1) * args.infer_batch_size * desc_per_image
            for ix in range(image_descriptors.size(0)):
                sample = np.random.choice(image_descriptors.size(1), desc_per_image, replace=False)
                startix = batchix + ix * desc_per_image
                descriptors[startix:startix + desc_per_image, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
    niter = 100
    kmeans = faiss.Kmeans(args.features_dim, args.num_clusters, niter=niter, verbose=False)
    kmeans.train(descriptors)
    return kmeans.centroids, descriptors