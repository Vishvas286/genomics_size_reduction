import sys
import torch

name = sys.argv[1]

model = torch.load(name)

model.sublayers[0].mask_flag = False
model.sublayers[1].layer.mask_flag = False
model.sublayers[2].mask_flag = False
model.sublayers[3].layer.mask_flag = False
model.sublayers[4].mask_flag = False
model.sublayers[5].layer.mask_flag = False
model.sublayers[6].mask_flag = False

model.sublayers[0].masks = dict()
model.sublayers[1].layer.masks = dict()
model.sublayers[2].masks = dict()
model.sublayers[3].layer.masks = dict()
model.sublayers[4].masks = dict()
model.sublayers[5].layer.masks = dict()
model.sublayers[6].masks = dict()

torch.save(model, sys.argv[2])
