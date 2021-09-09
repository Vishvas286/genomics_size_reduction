import torch
import sys
import models.mGru_flipflop as net

model1 = torch.load(sys.argv[1])            #From
model2 = torch.load("/mnt/2acd47df-cdc3-445d-b1ed-7669845d853d/project/taiyaki/models/standard.checkpoint")

model2.sublayers[0].conv.weight.data = model1.sublayers[0].conv.weight.data
model2.sublayers[0].conv.bias.data = model1.sublayers[0].conv.bias.data
model2.sublayers[1].layer.cudnn_gru.weight_ih_l0.data = model1.sublayers[1].layer.cudnn_gru.weight_ih_l0.data
model2.sublayers[1].layer.cudnn_gru.weight_hh_l0.data = model1.sublayers[1].layer.cudnn_gru.weight_hh_l0.data
model2.sublayers[1].layer.cudnn_gru.bias_ih_l0.data = model1.sublayers[1].layer.cudnn_gru.bias_ih_l0.data
model2.sublayers[1].layer.cudnn_gru.bias_hh_l0.data = model1.sublayers[1].layer.cudnn_gru.bias_hh_l0.data
model2.sublayers[2].cudnn_gru.weight_ih_l0.data = model1.sublayers[2].cudnn_gru.weight_ih_l0.data
model2.sublayers[2].cudnn_gru.weight_hh_l0.data = model1.sublayers[2].cudnn_gru.weight_hh_l0.data
model2.sublayers[2].cudnn_gru.bias_ih_l0.data = model1.sublayers[2].cudnn_gru.bias_ih_l0.data
model2.sublayers[2].cudnn_gru.bias_hh_l0.data = model1.sublayers[2].cudnn_gru.bias_hh_l0.data
model2.sublayers[3].layer.cudnn_gru.weight_ih_l0.data = model1.sublayers[3].layer.cudnn_gru.weight_ih_l0.data
model2.sublayers[3].layer.cudnn_gru.weight_hh_l0.data = model1.sublayers[3].layer.cudnn_gru.weight_hh_l0.data
model2.sublayers[3].layer.cudnn_gru.bias_ih_l0.data = model1.sublayers[3].layer.cudnn_gru.bias_ih_l0.data
model2.sublayers[3].layer.cudnn_gru.bias_hh_l0.data = model1.sublayers[3].layer.cudnn_gru.bias_hh_l0.data
model2.sublayers[4].cudnn_gru.weight_ih_l0.data = model1.sublayers[4].cudnn_gru.weight_ih_l0.data
model2.sublayers[4].cudnn_gru.weight_hh_l0.data = model1.sublayers[4].cudnn_gru.weight_hh_l0.data
model2.sublayers[4].cudnn_gru.bias_ih_l0.data = model1.sublayers[4].cudnn_gru.bias_ih_l0.data
model2.sublayers[4].cudnn_gru.bias_hh_l0.data = model1.sublayers[4].cudnn_gru.bias_hh_l0.data
model2.sublayers[5].layer.cudnn_gru.weight_ih_l0.data = model1.sublayers[5].layer.cudnn_gru.weight_ih_l0.data
model2.sublayers[5].layer.cudnn_gru.weight_hh_l0.data = model1.sublayers[5].layer.cudnn_gru.weight_hh_l0.data
model2.sublayers[5].layer.cudnn_gru.bias_ih_l0.data = model1.sublayers[5].layer.cudnn_gru.bias_ih_l0.data
model2.sublayers[5].layer.cudnn_gru.bias_hh_l0.data = model1.sublayers[5].layer.cudnn_gru.bias_hh_l0.data
model2.sublayers[6].linear.weight.data = model1.sublayers[6].linear.weight.data
model2.sublayers[6].linear.bias.data = model1.sublayers[6].linear.bias.data

torch.save(model2, "copied.checkpoint")