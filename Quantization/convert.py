import sys
import torch

name = sys.argv[1]

model = torch.load(name)

model.sublayers[0].q_flag = [False, False]
model.sublayers[0].run_flag = True
model.sublayers[0].q_data = dict()

model.sublayers[1].layer.q_flag = [False, False, False, False]
model.sublayers[1].layer.run_flag = True
model.sublayers[1].layer.q_data = dict()

model.sublayers[2].q_flag = [False, False, False, False]
model.sublayers[2].run_flag = True
model.sublayers[2].q_data = dict()

model.sublayers[3].layer.q_flag = [False, False, False, False]
model.sublayers[3].layer.run_flag = True
model.sublayers[3].layer.q_data = dict()

model.sublayers[4].q_flag = [False, False, False, False]
model.sublayers[4].run_flag = True
model.sublayers[4].q_data = dict()

model.sublayers[5].layer.q_flag = [False, False, False, False]
model.sublayers[5].layer.run_flag = True
model.sublayers[5].layer.q_data = dict()

model.sublayers[6].q_flag = [False, False]
model.sublayers[6].run_flag = True
model.sublayers[6].q_data = dict()

torch.save(model, sys.argv[2])