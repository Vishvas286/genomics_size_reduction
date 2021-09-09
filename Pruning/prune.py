import torch
import pickle
import math
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading model:", sys.argv[1], "\n")
if str(device) == "cuda:0":
    model = torch.load(sys.argv[1], map_location='cuda')
else:
    model = torch.load(sys.argv[1], map_location='cpu')

f = open("data.txt", 'r')
lines = f.readlines()
sublayers = list()
percentages = list()
f.close()

print("Pruning", int(len(lines)/2), "layer(s)")

for i in range(0, int(len(lines)), 2):
    sublayers.append(lines[i].strip())
    percentages.append(float(lines[i+1].strip()))

temp = list()


for i in range(int(len(lines)/2)):
    for name, param in model.named_parameters():
        if name == sublayers[i]:
            print("\n-------------------------------------------------------")
            print("Pruning layer:", name, "\nWith Sparsity:", percentages[i]*100, "%")
            rows = param.shape[0]
            cols = param.shape[1]
            mask = torch.ones([rows, cols])

            for x in range(0, rows):
                for y in range(0, cols):
                    temp.append(abs(param.data[x,y]))
            
            temp.sort()
            limit = temp[math.ceil(rows*cols*percentages[i])]

            for x in range(0, rows):
                for y in range(0, cols):
                    if abs(param.data[x,y]) < limit:
                        mask.data[x, y] = 0

            if str(device) == "cuda:0":
                mask = mask.cuda()
            mask = mask.to(dtype=torch.float32)

            s = sublayers[i].split(".")
            if len(s) == 5:
                if len(model.sublayers[int(s[1])].layer.weight_names) == 0:
                    model.sublayers[int(s[1])].layer.StoreMask([s[-1]], [mask])

                elif len(model.sublayers[int(s[1])].layer.weight_names) == 1:
                    if s[-1] in model.sublayers[int(s[1])].layer.weight_names:
                        model.sublayers[int(s[1])].layer.StoreMask([s[-1]], [mask])
                    else:
                        layer_name = model.sublayers[int(s[1])].layer.weight_names[0]
                        mask_temp = model.sublayers[int(s[1])].layer.masks[0]
                        model.sublayers[int(s[1])].layer.StoreMask([s[-1], layer_name], [mask, mask_temp])
                else:
                    index = model.sublayers[int(s[1])].layer.weight_names.index(s[-1])
                    if index == 0:
                        layer_name = model.sublayers[int(s[1])].layer.weight_names[1]
                        mask_temp = model.sublayers[int(s[1])].layer.masks[1]
                    elif index == 1:
                        layer_name = model.sublayers[int(s[1])].layer.weight_names[0]
                        mask_temp = model.sublayers[int(s[1])].layer.masks[0]

                    model.sublayers[int(s[1])].layer.StoreMask([s[-1], layer_name], [mask, mask_temp])
                    
                model.sublayers[int(s[1])].layer.ApplyMask()
            else:
                if len(model.sublayers[int(s[1])].weight_names) == 0:
                    model.sublayers[int(s[1])].StoreMask([s[-1]], [mask])

                elif len(model.sublayers[int(s[1])].weight_names) == 1:
                    if s[-1] in model.sublayers[int(s[1])].weight_names:
                        model.sublayers[int(s[1])].StoreMask([s[-1]], [mask])
                    else:
                        layer_name = model.sublayers[int(s[1])].weight_names[0]
                        mask_temp = model.sublayers[int(s[1])].masks[0]
                        model.sublayers[int(s[1])].StoreMask([s[-1], layer_name], [mask, mask_temp])
                else:
                    index = model.sublayers[int(s[1])].weight_names.index(s[-1])
                    if index == 0:
                        layer_name = model.sublayers[int(s[1])].weight_names[1]
                        mask_temp = model.sublayers[int(s[1])].masks[1]
                    elif index == 1:
                        layer_name = model.sublayers[int(s[1])].weight_names[0]
                        mask_temp = model.sublayers[int(s[1])].masks[0]

                    model.sublayers[int(s[1])].StoreMask([s[-1], layer_name], [mask, mask_temp])
                    
                model.sublayers[int(s[1])].ApplyMask()
            

            print("-------------------------------------------------------\n")


torch.save(model, 'pruned.checkpoint')
print("Model Saved as pruned.checkpoint")
