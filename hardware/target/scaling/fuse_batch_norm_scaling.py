import numpy as np


# random fp array
N = 64
random_array = np.random.rand(64, 64) * 2 - 1

# int8 quantized array
scale = 255 / (np.max(random_array) - np.min(random_array))
zero_point = 0
quant_array = np.round(random_array * scale + zero_point)

print("random_array", random_array)
print("quant_array", quant_array)
print("scale", scale)


# dequantize
dequant_array = (quant_array - zero_point) / scale

print("dequant_array", dequant_array)

# batch norm
mean = np.mean(dequant_array, axis=0)
var = np.var(dequant_array, axis=0)
epsilon = 1e-5
bn_array = (dequant_array - mean) / np.sqrt(var + epsilon)
print("bn_array", bn_array)

# fused
fused_subtraction = (zero_point - scale * mean) / (scale * np.sqrt(var + epsilon))
fused_division = 1 / (scale * np.sqrt(var + epsilon))
fused_bn_array = quant_array * fused_division + fused_subtraction
print("fused_bn_array", fused_bn_array)
