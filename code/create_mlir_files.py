import os

class MlirFiles():
    def __init__(self, args, layers):
        self.layers = layers
        self.model_name, _ = os.path.splitext(os.path.basename(args.read_mlir))
        print(self.model_name)
        self.permute = args.permute
        self.tile = args.tile
        self.unroll = args.unroll
        if self.permute:
            self.loop_optimizer = 'permute'
        elif self.tile:
            self.loop_optimizer = 'tile'
        elif self.unroll:
            self.loop_optimizer = 'unroll'

    def create_mlir_function(self, layer_name, current_layer):
        if layer_name.startswith("conv2d"):
            if current_layer.clipped_output_channel is not None:
                output_channel = current_layer.clipped_output_channel
            else:
                output_channel = current_layer.output_channel
            if current_layer.clipped_input_channel is not None:
                input_channel = current_layer.clipped_input_channel
            else:
                input_channel = current_layer.input_channel
            input_shape = "x".join([str(current_layer.input_batch),
                                    str(current_layer.input_width),
                                    str(current_layer.input_height),
                                    str(input_channel)])
            kernel_shape = "x".join([str(current_layer.kernel_width),
                                     str(current_layer.kernel_height),
                                     str(input_channel),
                                     str(output_channel)])
            output_shape = "x".join([str(current_layer.output_batch),
                                     str(current_layer.output_width),
                                     str(current_layer.output_height),
                                     str(output_channel)])
            dilations = current_layer.dilations
            strides = current_layer.strides
            linalg_name = "conv_2d_nhwc_hwcf"
            linalg_line = f"    linalg.{linalg_name} {{dilations = dense<{dilations}> : tensor<2xi64>, strides = dense<{strides}> : tensor<2xi64>}}"
            
        elif layer_name.startswith("depthwise_conv2d"):
            if current_layer.clipped_input_channel is not None:
                input_channel = current_layer.clipped_input_channel
            else:
                input_channel = current_layer.input_channel
            input_shape = "x".join([str(current_layer.input_batch),
                                    str(current_layer.input_width),
                                    str(current_layer.input_height),
                                    str(input_channel)])
            kernel_shape = "x".join([str(current_layer.kernel_width),
                                     str(current_layer.kernel_height),
                                     str(input_channel)])
            output_shape = "x".join([str(current_layer.output_batch),
                                     str(current_layer.output_width),
                                     str(current_layer.output_height),
                                     str(input_channel)])
            dilations = current_layer.dilations
            strides = current_layer.strides
            linalg_name = "depthwise_conv_2d_nhwc_hwc"
            linalg_line = f"    linalg.{linalg_name} {{dilations = dense<{dilations}> : tensor<2xi64>, strides = dense<{strides}> : tensor<2xi64>}}"
        
        elif layer_name.startswith("matmul"):
            if current_layer.clipped_output_width is not None:
                output_width = current_layer.clipped_output_width
            else:
                output_width = current_layer.output_width
            if current_layer.clipped_output_height is not None:
                output_height = current_layer.clipped_output_height
            else:
                output_height = current_layer.output_height
            if current_layer.clipped_kernel_width is not None:
                kernel_width = current_layer.clipped_kernel_width
            else:
                kernel_width = current_layer.kernel_width
            if current_layer.clipped_kernel_height is not None:
                kernel_height = current_layer.clipped_kernel_height
            else:
                kernel_height = current_layer.kernel_height
            if current_layer.clipped_input_height is not None:
                input_height = current_layer.clipped_input_height
            else:
                input_height = current_layer.input_height
            if current_layer.clipped_input_width is not None:
                input_width = current_layer.clipped_input_width
            else:
                input_width = current_layer.input_width
            input_shape = "x".join([str(current_layer.input_batch),
                                    str(input_width),
                                    str(input_height)])
            kernel_shape = "x".join([str(current_layer.kernel_batch),
                                     str(kernel_width),
                                     str(kernel_height)])
            output_shape = "x".join([str(current_layer.output_batch),
                                     str(output_width),
                                     str(output_height)])
            linalg_name = "batch_matmul"
            linalg_line = f"    linalg.{linalg_name}"
        self.write_mlir_file(layer_name, linalg_line, input_shape, kernel_shape, output_shape)

    def write_mlir_file(self, layer_name, linalg_line, input_shape, kernel_shape, output_shape):
        # Assuming self.layers[layer_name].file_name is already defined
        file_path = f"layers_{self.model_name}_{self.loop_optimizer}/{layer_name}.mlir"
        self.layers[layer_name].file_path = file_path
        # Get the directory name
        directory = os.path.dirname(file_path)
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w', encoding='UTF-8') as file:
            file.write(f"func.func @main(%arg0: memref<{input_shape}xf32>, ")
            file.write(f"%arg1: memref<{kernel_shape}xf32>, ")
            file.write(f"%arg2: memref<{output_shape}xf32>) {{\n")
            file.write("  cf.br ^bb1\n")
            file.write("^bb1:  // pred: ^bb0\n")
            file.write("  soda.launch {\n")
            # Modify dilations and strides line to use direct variable substitution
            file.write(f"{linalg_line}")
            file.write(f" ins(%arg0, %arg1 : memref<{input_shape}xf32>, ")
            file.write(f"memref<{kernel_shape}xf32>) ")
            file.write(f"outs(%arg2 : memref<{output_shape}xf32>)\n")
            file.write("    soda.terminator\n")
            file.write("  }\n")
            file.write("  return\n")
            file.write("}\n")

    def execute(self):
        for layer_name, layer in self.layers.items():
            self.create_mlir_function(layer_name, layer)
        return self.layers
