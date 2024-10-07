import math

class DepthwiseDepthConv2D:
    def __init__(self, args, input_tensor_shape, kernel_tensor_shape, output_tensor_shape, dilations, strides):
        
        self.permute = args.permute
        self.tile = args.tile
        self.unroll = args.unroll

        self.file_path = None

        self.dilations = dilations
        self.strides = strides
        print("---------------------------------------------------")
        print(f"DepthConv2D: Dilations: {self.dilations}")
        print(f"DepthConv2D: Strides: {self.strides}")


        self.input_batch = input_tensor_shape[0]
        self.input_width = input_tensor_shape[1]
        self.input_height = input_tensor_shape[2]
        self.input_channel = input_tensor_shape[3]

        print(f"DepthConv2D: Input batch size: {self.input_batch}")
        print(f"DepthConv2D: Input width: {self.input_width}")
        print(f"DepthConv2D: Input height: {self.input_height}")
        print(f"DepthConv2D: Input channels: {self.input_channel}")

        self.output_batch = output_tensor_shape[0]
        self.output_width = output_tensor_shape[1]
        self.output_height = output_tensor_shape[2]
        self.output_channel = output_tensor_shape[3]
        self.output_multiplier = None
        if len(output_tensor_shape) == 5:
            self.output_multiplier = output_tensor_shape[4]

        print(f"DepthConv2D: Output batch size: {self.output_batch}")
        print(f"DepthConv2D: Output width: {self.output_width}")
        print(f"DepthConv2D: Output height: {self.output_height}")
        print(f"DepthConv2D: Output channels: {self.output_channel}")

        self.kernel_width = kernel_tensor_shape[0]
        self.kernel_height = kernel_tensor_shape[1]
        self.kernel_input_channel = kernel_tensor_shape[2]
        self.kernel_multiplier = None
        if len(kernel_tensor_shape) == 4:
            self.kernel_multiplier = kernel_tensor_shape[3]

        print(f"DepthConv2D: Kernel width: {self.kernel_width}")
        print(f"DepthConv2D: Kernel height: {self.kernel_height}")
        print(f"DepthConv2D: Kernel input channels: {self.kernel_input_channel}")
        print(f"DepthConv2D: Kernel multiplier: {self.kernel_multiplier}")
        
        self.flop_count = self.output_batch * self.output_width * self.output_height * self.kernel_width * self.kernel_height * self.input_channel * 2 

        self.clipped_input_channel = None

        self.no_of_tiles = 1
        
        if self.kernel_multiplier is not None:
            self.no_of_tiles *= self.kernel_multiplier

        if self.tile or self.permute:
            if self.input_height <= 16 and self.input_channel > 128:
                factors = self.get_factors(self.input_channel)
                factors_list = [factor for factor in factors if factor <= 128]
                self.clipped_input_channel = factors_list[-1]
                self.no_of_tiles *= self.input_channel/self.clipped_input_channel
            elif self.input_height > 16 and self.input_height <= 32 and self.input_channel > 32:
                factors = self.get_factors(self.input_channel)
                factors_list = [factor for factor in factors if factor <= 32]
                self.clipped_input_channel = factors_list[-1]
                self.no_of_tiles *= self.input_channel/self.clipped_input_channel
            elif self.input_height > 32 and self.input_height <= 64 and self.input_channel > 8:
                factors = self.get_factors(self.input_channel)
                factors_list = [factor for factor in factors if factor <= 8]
                self.clipped_input_channel = factors_list[-1]
                self.no_of_tiles *= self.input_channel/self.clipped_input_channel
            elif self.input_height > 64 and self.input_channel > 1:
                self.clipped_input_channel = 1
                self.no_of_tiles *= self.input_channel/self.clipped_input_channel
        elif self.unroll:
            if self.input_height <= 64:
                if self.kernel_height <= 3 and self.input_channel > 32:               
                    factors = self.get_factors(self.input_channel)
                    factors_list = [factor for factor in factors if factor <= 32]
                    self.clipped_input_channel = factors_list[-1]
                    self.no_of_tiles *= self.input_channel/self.clipped_input_channel
                elif self.kernel_height > 3 and self.kernel_height <= 5 and self.input_channel > 8:               
                    factors = self.get_factors(self.input_channel)
                    factors_list = [factor for factor in factors if factor <= 8]
                    self.clipped_input_channel = factors_list[-1]
                    self.no_of_tiles *= self.input_channel/self.clipped_input_channel
                elif self.kernel_height > 5 and self.kernel_height <= 7 and self.input_channel > 4:               
                    factors = self.get_factors(self.input_channel)
                    factors_list = [factor for factor in factors if factor <= 4]
                    self.clipped_input_channel = factors_list[-1]
                    self.no_of_tiles *= self.input_channel/self.clipped_input_channel
                elif self.kernel_height > 7 and self.kernel_height <= 11 and self.input_channel > 1:               
                    self.clipped_input_channel = 1
                    self.no_of_tiles *= self.input_channel/self.clipped_input_channel
            elif self.input_height > 64 and self.input_channel > 1:
                self.clipped_input_channel = 1
                self.no_of_tiles *= self.input_channel/self.clipped_input_channel

    
    def get_factors(self, n):
        """Return the factors of n."""
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append(i)
        return factors