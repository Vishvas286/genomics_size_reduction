<h1> Size Reduction of base calling model</h1>
This repository contains all the necessary files to perform pruning and quantization on the Taiyaki base calling models.
<br><br>

First clone [Taiyaki](https://github.com/nanoporetech/taiyaki) repository. Details about setting up taiyaki and training the models are also available there. Once taiyaki is set up, you can proceed with the steps below.

Due to PyTorch version incompatibility, the pre-trained model (r941_dna_minion.checkpoint) from taiyaki can't be used for quantization. We made a replica of that model  which removes this incompatibility. This updated model is named original.checkpoint. 

<br><br>
<h2>Pruning</h2>

The files to perform pruning are present in the "Pruning" folder. The main script to be run is prune.py. It takes one argument which is the checkpoint file to be pruned. The layers to be pruned and the sparsity for each layer can be mentioned in data.txt. The first line of data.txt file should be the name of the layer and the second line should be the sparsity (in the range 0-1) you want to achieve for the layer in the above line. You can look at the data.txt file present in this repository as a exampmle. Please note that the file should have only even number of lines.
<br><br>
Before performing pruning there are two steps to be performed. First the script "convert_p.py" has to be run with two arguments. One is the model checkpoint and the second is the name of the output file.

Second, the contents of "layers.py" file under taiyaki/taiyaki needs to be replaced with the contents of "layers_p.py" from this repository. This enables the model to make use of the mask during training or retraining.

<br>


```
python convert_p.py original.checkpoint input_to_prune.checkpoint
python prune.py input_to_prune.checkpoint
```


<br><br>
<h2>Quantization</h2>

The files used to perform Quantization are present under "Quantization" folder. The main script to run is "scale_quant.py" to do 8-bit quantization. It takes two arguments where the second argument is optional. The first argument is the checkpoint file to be quantized and the second argument should be "all" only if you want to quantize all the layers of the model. If you want to quantize only some specific layers then the name of those layers should be mentioned in data.txt file.
<br><br>
Before performing Quantization first step is to run "convert_q.py" which takes two arguments. The first argument is the model checkpoint file and the second argument is the output checkpoint file.
The second step is to replace the contents of "layers.py" file under taiyaki/taiyaki with the contents of "layers_q.py" from our repository.

One important parameter in "scale_quant.py" is [float_format](https://github.com/Vishvas286/genomics_size_reduction/blob/922a6b2180479e8fe95d85a73b15c29b888fef5b/Quantization/scale_quant.py#L332). If this is set to false, then when quantization takes place and all the weights are converted to the range (-127, 128) and the data type of the weights are also made as torch.int8.

Setting float_format = True will cause the script to convert the values to the range (-127, 128) and then dequantize them to float format again. Now even though the weights are in floating point format, they can be represented using only 8-bits instead of 32-bit.


<br>

```
python convert_q.py original.checkpoint input_to_quantize.checkpoint
python scale_quant.py input_to_quantize.checkpoint all
```

Some more variables to take note of are 'q_data' and 'q_flag'. The q_data variable is a dictionary with the keys as the layer which has been quantized and the value is a list with values [scale, zeropoint] for that layer. The q_flag variable is a list with 4 values if the layer is GRU and 2 values if the layer is Convolution or Linear. 

```
For GRU the flags correspond to -> 
[weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0]

For convolution and linear ->
[weight, bias]
```
If a value in the list is True, it means the corresponding tensor has been quantized. 

Lastly, two variables named 'convert_flag' and 'run_flag' are present in layers_q.py. 

The 'run_flag' is set to True. When it is True, it means no input has been passed to the model yet. After we start giving the input to the model, the flag will be set to False. (Note that convert_flag also has to be True, or else the value of run_flag doesn't matter).

'convert_flag' is used when the tensor is stored in 8-bit format but it needs to be converted back to 32-bits before starting base calling. When it is set to True, and the first input arrives to the model, all the int8 values will be dequantized and replaced with 32-bit values. When the second input arrives, 'run_flag' will be False and dequantization doesn't take place. (This is because dequantization is a one time operation).

<br><br>
<h2>Analysis</h2>

The steps to perform assembly and polishing are mentioned in [this](https://github.com/rrwick/Basecalling-comparison) repository. The reference fasta file that we used has been put in our repository.


<br><br>
<h2>Variant Calling</h2>

We used a variant caller called Clair. Steps to install Clair are mentioned [here](https://github.com/HKU-BAL/Clair). For our project we used the latest version which is [Clair3](https://github.com/HKU-BAL/Clair3). We ran the clair3 script to produce the vcf files. 

Once the vcf files are obtained, for evaluation you can use a python script called "hap.py". Details about "hap.py" are mentioned [here](https://github.com/Illumina/hap.py). 

The Clair3 repository also has details about how to use intsall the required tools for running evaluation using "hap.py". 

