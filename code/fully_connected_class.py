class FullyConnected:
    def __init__(self, args, input_tensor_shape, kernel_tensor_shape, output_tensor_shape):

        self.permute = args.permute
        self.tile = args.tile
        self.unroll = args.unroll
        
        self.file_path = None

        self.input_batch = input_tensor_shape[0]
        self.input_width = input_tensor_shape[1]
        self.input_height = input_tensor_shape[2]

        print("---------------------------------------------------")
        print(f"FullyConnected: Input batch size: {self.input_batch}")
        print(f"FullyConnected: Input width: {self.input_width}")
        print(f"FullyConnected: Input height: {self.input_height}")

        self.output_batch = output_tensor_shape[0]
        self.output_width = output_tensor_shape[1]
        self.output_height = output_tensor_shape[2]

        print(f"FullyConnected: Output batch size: {self.output_batch}")
        print(f"FullyConnected: Output width: {self.output_width}")
        print(f"FullyConnected: Output height: {self.output_height}")

        self.kernel_batch = kernel_tensor_shape[0]
        self.kernel_width = kernel_tensor_shape[1]
        self.kernel_height = kernel_tensor_shape[2]

        print(f"FullyConnected: Kernel batch size: {self.kernel_batch}")
        print(f"FullyConnected: Kernel width: {self.kernel_width}")
        print(f"FullyConnected: Kernel height: {self.kernel_height}")

        self.flop_count = self.output_batch * self.output_width * self.output_height * self.kernel_width * 2 

        self.clipped_output_width = None
        self.clipped_output_height = None
        self.clipped_input_width = None
        self.clipped_input_height = None
        self.clipped_kernel_width= None
        self.clipped_kernel_height = None

        self.no_of_tiles = 1

        if self.tile or self.permute:
            if self.output_width > 128:
                factors = self.get_factors(self.output_width)
                factors_list = [factor for factor in factors if factor <= 128]
                self.clipped_output_width = factors_list[-1]
                self.clipped_input_width = factors_list[-1]
                print(f"FullyConnected: Clipped output width to {self.clipped_output_width} due to output_width > 128")
                print(f"FullyConnected: Clipped input width to {self.clipped_input_width} due to input_width > 128")
                self.no_of_tiles *= self.output_width/self.clipped_output_width
            if self.output_height > 128:
                factors = self.get_factors(self.output_height)
                factors_list = [factor for factor in factors if factor <= 128]
                self.clipped_output_height = factors_list[-1]
                self.clipped_kernel_height = factors_list[-1]
                print(f"FullyConnected: Clipped output height to {self.clipped_output_height} due to output_height > 128")
                print(f"FullyConnected: Clipped kernel height to {self.clipped_kernel_height} due to kernel_height > 128")
                self.no_of_tiles *= self.output_height/self.clipped_output_height
            if self.kernel_width == self.input_height and self.kernel_width > 128 and self.input_height > 128:
                factors = self.get_factors(self.kernel_width)
                factors_list = [factor for factor in factors if factor <= 128]
                self.clipped_kernel_width = factors_list[-1]
                self.clipped_input_height = factors_list[-1]
                print(f"FullyConnected: Clipped kernel width to {self.clipped_kernel_width} due to kernel_width > 128")
                print(f"FullyConnected: Clipped input height to {self.clipped_input_height} due to input_height > 128")
                self.no_of_tiles *= self.kernel_width/self.clipped_kernel_width
        elif self.unroll:
            if self.output_width > 32:
                factors = self.get_factors(self.output_width)
                factors_list = [factor for factor in factors if factor <= 32]
                self.clipped_output_width = factors_list[-1]
                self.clipped_input_width = factors_list[-1]
                print(f"FullyConnected: Clipped output width to {self.clipped_output_width} due to output_width > 32")
                print(f"FullyConnected: Clipped input width to {self.clipped_input_width} due to input_width > 32")
                self.no_of_tiles *= self.output_width/self.clipped_output_width
            if self.output_height > 32:
                factors = self.get_factors(self.output_height)
                factors_list = [factor for factor in factors if factor <= 32]
                self.clipped_output_height = factors_list[-1]
                self.clipped_kernel_height = factors_list[-1]
                print(f"FullyConnected: Clipped output height to {self.clipped_output_height} due to output_height > 32")
                print(f"FullyConnected: Clipped kernel height to {self.clipped_kernel_height} due to kernel_height > 32")
                self.no_of_tiles *= self.output_height/self.clipped_output_height
            if self.kernel_width == self.input_height and self.kernel_width > 32 and self.input_height > 32:
                factors = self.get_factors(self.kernel_width)
                factors_list = [factor for factor in factors if factor <= 32]
                self.clipped_kernel_width = factors_list[-1]
                self.clipped_input_height = factors_list[-1]
                print(f"FullyConnected: Clipped kernel width to {self.clipped_kernel_width} due to kernel_width > 32")
                print(f"FullyConnected: Clipped input height to {self.clipped_input_height} due to input_height > 32")
                self.no_of_tiles *= self.kernel_width/self.clipped_kernel_width
    def get_factors(self, n):
        """Return the factors of n."""
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append(i)
        return factors
