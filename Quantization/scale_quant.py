import collections
import torch
import sys

QTensor = collections.namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
    # Calc Scale and zero point of next 
    qmin = -2.**(num_bits - 1)
    qmax = 2.**(num_bits - 1) - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point

def quantize_tensor(x, num_bits):
    if x.sum() == 0:
        return QTensor(tensor=x, scale=0, zero_point=0)
    qmin = -2.**(num_bits - 1)
    qmax = 2.**(num_bits - 1) - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().type(torch.float32)

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

def main(num_bits, float_format=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print()
    print("Loading model:", sys.argv[1], "\n")
    if str(device) == "cuda:0":
        model = torch.load(sys.argv[1], map_location='cuda')
    else:
        model = torch.load(sys.argv[1], map_location='cpu')

    q = list()
    i = 0

    try:
        if sys.argv[2] == "all":
            for name, param in model.named_parameters():
                q.append(name)
        else:
            raise RuntimeError("Invalid parameters")

    except IndexError:
        f = open("data.txt", 'r')
        lines = f.readlines()
        f.close()

        print("Quantizing", int(len(lines)), "layer(s):")

        for ind in range(int(len(lines))):
            q.append(lines[ind].strip())

    for layer in q:
        for name, param in model.named_parameters():
            if layer in name:
                i += 1
                s = layer.split(".")
                print("Quantizing layer:", layer)
                if "cudnn_gru" in s and len(s) == 5:
                    if "weight_ih_l0" in s:
                        model.sublayers[int(s[1])].layer.q_flag[0] = True
                    elif "weight_hh_l0" in s:
                        model.sublayers[int(s[1])].layer.q_flag[1] = True
                    elif "bias_ih_l0" in s:
                        model.sublayers[int(s[1])].layer.q_flag[2] = True
                    elif "bias_hh_l0" in s:
                        model.sublayers[int(s[1])].layer.q_flag[3] = True
                    model.sublayers[int(s[1])].layer.run_flag = False
                    model.sublayers[int(s[1])].layer.convert_flag = False

                    sd = model.sublayers[int(s[1])].layer.cudnn_gru.state_dict()
                    for n, _ in model.sublayers[int(s[1])].layer.cudnn_gru.named_parameters():
                        if n == s[-1]:
                            temp = quantize_tensor(sd[s[-1]].data, num_bits=num_bits)
                            if float_format:
                                sd[s[-1]].data = torch.nn.Parameter(dequantize_tensor(temp))
                            else:
                                sd[s[-1]].data = temp.tensor
                            
                    model.sublayers[int(s[1])].layer.cudnn_gru.load_state_dict(sd)
                    model.sublayers[int(s[1])].layer.q_data[s[-1]] = [temp.scale, temp.zero_point]

                elif "cudnn_gru" in s and len(s) == 4:
                    if "weight_ih_l0" in s:
                        model.sublayers[int(s[1])].q_flag[0] = True
                    elif "weight_hh_l0" in s:
                        model.sublayers[int(s[1])].q_flag[1] = True
                    elif "bias_ih_l0" in s:
                        model.sublayers[int(s[1])].q_flag[2] = True
                    elif "bias_hh_l0" in s:
                        model.sublayers[int(s[1])].q_flag[3] = True
                    model.sublayers[int(s[1])].run_flag = False
                    model.sublayers[int(s[1])].convert_flag = False

                    sd = model.sublayers[int(s[1])].cudnn_gru.state_dict()
                    for n, _ in model.sublayers[int(s[1])].cudnn_gru.named_parameters():
                        if n == s[-1]:
                            temp = quantize_tensor(sd[s[-1]].data, num_bits=num_bits)
                            if float_format:
                                sd[s[-1]].data = torch.nn.Parameter(dequantize_tensor(temp))
                            else:
                                sd[s[-1]].data = temp.tensor
                            
                    model.sublayers[int(s[1])].cudnn_gru.load_state_dict(sd)
                    model.sublayers[int(s[1])].q_data[s[-1]] = [temp.scale, temp.zero_point]
                elif "conv" in s:
                    if "weight" in s:
                        model.sublayers[int(s[1])].q_flag[0] = True
                    elif "bias" in s:
                        model.sublayers[int(s[1])].q_flag[1] = True
                    model.sublayers[int(s[1])].run_flag = False
                    model.sublayers[int(s[1])].convert_flag = False

                    sd = model.sublayers[int(s[1])].conv.state_dict()
                    for n, _ in model.sublayers[int(s[1])].conv.named_parameters():
                        if n == s[-1]:
                            temp = quantize_tensor(sd[s[-1]].data, num_bits=num_bits)
                            if float_format:
                                sd[s[-1]].data = torch.nn.Parameter(dequantize_tensor(temp))
                            else:
                                sd[s[-1]].data = temp.tensor

                    model.sublayers[int(s[1])].conv.load_state_dict(sd)
                    model.sublayers[int(s[1])].q_data[s[-1]] = [temp.scale, temp.zero_point]
                elif "linear" in s:
                    if "weight" in s:
                        model.sublayers[int(s[1])].q_flag[0] = True
                    elif "bias" in s:
                        model.sublayers[int(s[1])].q_flag[1] = True
                    model.sublayers[int(s[1])].run_flag = False
                    model.sublayers[int(s[1])].convert_flag = False

                    sd = model.sublayers[int(s[1])].linear.state_dict()
                    for n, _ in model.sublayers[int(s[1])].linear.named_parameters():
                        if n == s[-1]:
                            temp = quantize_tensor(sd[s[-1]].data, num_bits=num_bits)
                            if float_format:
                                sd[s[-1]].data = torch.nn.Parameter(dequantize_tensor(temp))
                            else:
                                sd[s[-1]].data = temp.tensor

                    model.sublayers[int(s[1])].linear.load_state_dict(sd)
                    model.sublayers[int(s[1])].q_data[s[-1]] = [temp.scale, temp.zero_point]
                else:
                    raise RuntimeError("No quantization function available for the given layer")
    if not float_format:
        for layer in q:
            if layer == "sublayers.0.conv.weight":
                z = model.sublayers[0].conv.weight
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[0].conv.weight = z

            elif layer == "sublayers.0.conv.bias":
                z = model.sublayers[0].conv.bias
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[0].conv.bias = z

            elif layer == "sublayers.1.layer.cudnn_gru.weight_ih_l0":
                z = model.sublayers[1].layer.cudnn_gru.weight_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[1].layer.cudnn_gru.weight_ih_l0 = z

            elif layer == "sublayers.1.layer.cudnn_gru.weight_hh_l0":
                z = model.sublayers[1].layer.cudnn_gru.weight_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[1].layer.cudnn_gru.weight_hh_l0 = z

            elif layer == "sublayers.1.layer.cudnn_gru.bias_ih_l0":
                z = model.sublayers[1].layer.cudnn_gru.bias_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[1].layer.cudnn_gru.bias_ih_l0 = z

            elif layer == "sublayers.1.layer.cudnn_gru.bias_hh_l0":
                z = model.sublayers[1].layer.cudnn_gru.bias_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[1].layer.cudnn_gru.bias_hh_l0 = z

            elif layer == "sublayers.2.cudnn_gru.weight_ih_l0":
                z = model.sublayers[2].cudnn_gru.weight_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[2].cudnn_gru.weight_ih_l0 = z

            elif layer == "sublayers.2.cudnn_gru.weight_hh_l0":
                z = model.sublayers[2].cudnn_gru.weight_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[2].cudnn_gru.weight_hh_l0 = z

            elif layer == "sublayers.2.cudnn_gru.bias_ih_l0":
                z = model.sublayers[2].cudnn_gru.bias_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[2].cudnn_gru.bias_ih_l0 = z

            elif layer == "sublayers.2.cudnn_gru.bias_hh_l0":
                z = model.sublayers[2].cudnn_gru.bias_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[2].cudnn_gru.bias_hh_l0 = z

            elif layer == "sublayers.3.layer.cudnn_gru.weight_ih_l0":
                z = model.sublayers[3].layer.cudnn_gru.weight_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[3].layer.cudnn_gru.weight_ih_l0 = z

            elif layer == "sublayers.3.layer.cudnn_gru.weight_hh_l0":
                z = model.sublayers[3].layer.cudnn_gru.weight_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[3].layer.cudnn_gru.weight_hh_l0 = z

            elif layer == "sublayers.3.layer.cudnn_gru.bias_ih_l0":
                z = model.sublayers[3].layer.cudnn_gru.bias_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[3].layer.cudnn_gru.bias_ih_l0 = z

            elif layer == "sublayers.3.layer.cudnn_gru.bias_hh_l0":
                z = model.sublayers[3].layer.cudnn_gru.bias_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[3].layer.cudnn_gru.bias_hh_l0 = z

            elif layer == "sublayers.4.cudnn_gru.weight_ih_l0":
                z = model.sublayers[4].cudnn_gru.weight_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[4].cudnn_gru.weight_ih_l0 = z

            elif layer == "sublayers.4.cudnn_gru.weight_hh_l0":
                z = model.sublayers[4].cudnn_gru.weight_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[4].cudnn_gru.weight_hh_l0 = z

            elif layer == "sublayers.4.cudnn_gru.bias_ih_l0":
                z = model.sublayers[4].cudnn_gru.bias_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[4].cudnn_gru.bias_ih_l0 = z

            elif layer == "sublayers.4.cudnn_gru.bias_hh_l0":
                z = model.sublayers[4].cudnn_gru.bias_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[4].cudnn_gru.bias_hh_l0 = z
            
            elif layer == "sublayers.5.layer.cudnn_gru.weight_ih_l0":
                z = model.sublayers[5].layer.cudnn_gru.weight_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[5].layer.cudnn_gru.weight_ih_l0 = z

            elif layer == "sublayers.5.layer.cudnn_gru.weight_hh_l0":
                z = model.sublayers[5].layer.cudnn_gru.weight_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[5].layer.cudnn_gru.weight_hh_l0 = z

            elif layer == "sublayers.5.layer.cudnn_gru.bias_ih_l0":
                z = model.sublayers[5].layer.cudnn_gru.bias_ih_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[5].layer.cudnn_gru.bias_ih_l0 = z

            elif layer == "sublayers.5.layer.cudnn_gru.bias_hh_l0":
                z = model.sublayers[5].layer.cudnn_gru.bias_hh_l0
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[5].layer.cudnn_gru.bias_hh_l0 = z

            elif layer == "sublayers.6.linear.weight":
                z = model.sublayers[6].linear.weight
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[6].linear.weight = z

            elif layer == "sublayers.6.linear.bias":
                z = model.sublayers[6].linear.bias
                z = z.type(torch.int8)
                z = torch.nn.Parameter(z, requires_grad=False)
                model.sublayers[6].linear.bias = z

    if i != len(q):
        raise RuntimeError("Mistake in list of layers")

    torch.save(model, "quantized_model.checkpoint")

if __name__ == "__main__":
    num_bits = 8
    main(num_bits=num_bits, float_format=True)