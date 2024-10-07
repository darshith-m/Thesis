import re
from conv2d_class import Conv2D
from depthwise_conv2d_class import DepthwiseDepthConv2D
from fully_connected_class import FullyConnected

regex_patterns = {
    "conv2d": r'linalg\.conv_2d_nhwc_hwcf\s*\{\s*dilations\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*,\s*strides\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*[^}]*\}\s*ins\(\s*([^:]+):\s*memref<([^>]+)>\s*,\s*memref<([^>]+)>s*\)\s*outs\(\s*([^:]+):\s*memref<([^>]+)>\s*\)',
    "depthwise_conv2d_multiplier": r'linalg\.depthwise_conv_2d_nhwc_hwcm\s*\{\s*dilations\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*,\s*strides\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*[^}]*\}\s*ins\(\s*([^:]+):\s*memref<([^>]+)>\s*,\s*memref<([^>]+)>s*\)\s*outs\(\s*([^:]+):\s*memref<([^>]+)>\s*\)',
    "depthwise_conv2d": r'linalg\.depthwise_conv_2d_nhwc_hwc\s*\{\s*dilations\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*,\s*strides\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*[^}]*\}\s*ins\(\s*([^:]+):\s*memref<([^>]+)>\s*,\s*memref<([^>]+)>s*\)\s*outs\(\s*([^:]+):\s*memref<([^>]+)>\s*\)',
    "matmul": r'linalg\.batch_matmul\s*ins\(\s*([^:]+):\s*memref<([^>]+)>\s*,\s*memref<([^>]+)>s*\)\s*outs\(\s*([^:]+):\s*memref<([^>]+)>\s*\)'
}

def process_tensor_shape(tensor_shape):
    dimensions, _ = tensor_shape.strip().rsplit('x', 1)
    return list(map(int, dimensions.split('x')))

def read_file(args):
    layers = {}
    with open(args.read_mlir, 'r', encoding="UTF-8") as file:
        conv2d_count = 0
        depthwise_conv2d_count = 0
        matmul_count = 0
        for line in file:
            # Loop through each regex pattern in the dictionary
            for name, pattern in regex_patterns.items():
                match = re.findall(pattern, line)  # Find all matches for the current line
                if match and name == "conv2d" and args.conv2d:  # If there are matches
                    dilations = int(match[0][0])
                    strides = int(match[0][1])
                    input_tensor_shape = process_tensor_shape(match[0][3])
                    kernel_tensor_shape = process_tensor_shape(match[0][4])
                    output_tensor_shape = process_tensor_shape(match[0][6])
                    layer = Conv2D(args, input_tensor_shape, kernel_tensor_shape, output_tensor_shape, dilations, strides)
                    conv2d_count += 1
                    layers[f"{name}_{conv2d_count}"] = layer
                elif match and (name == "depthwise_conv2d" or name == "depthwise_conv2d_multiplier") and args.depthwise_conv2d:
                    dilations = int(match[0][0])
                    strides = int(match[0][1])
                    input_tensor_shape = process_tensor_shape(match[0][3])
                    kernel_tensor_shape = process_tensor_shape(match[0][4])
                    output_tensor_shape = process_tensor_shape(match[0][6])
                    layer = DepthwiseDepthConv2D(args, input_tensor_shape, kernel_tensor_shape, output_tensor_shape, dilations, strides)
                    depthwise_conv2d_count += 1
                    layers[f"{name}_{depthwise_conv2d_count}"] = layer
                elif match and name == "matmul" and args.matmul:
                    input_tensor_shape = process_tensor_shape(match[0][1])
                    kernel_tensor_shape = process_tensor_shape(match[0][2])
                    output_tensor_shape = process_tensor_shape(match[0][4])
                    layer = FullyConnected(args, input_tensor_shape, kernel_tensor_shape, output_tensor_shape)
                    matmul_count += 1
                    layers[f"{name}_{matmul_count}"] = layer
    print(f"Conv2d count: {conv2d_count}")
    print(f"Depthwise_Conv2d count: {depthwise_conv2d_count}")
    print(f"Matmul count: {matmul_count}")
    return layers
