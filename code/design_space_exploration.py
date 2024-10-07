import os
import re
import csv
import subprocess
import numpy as np
from itertools import permutations

class DSE:
    def __init__(self, args, layers):
        
        self.model_name, _ = os.path.splitext(os.path.basename(args.read_mlir))
        self.layers = layers
        self.permute = args.permute
        self.tile = args.tile
        self.unroll = args.unroll

        self.commands = None
        self.current_configuration = None
        self.current_layer_name = None

        if self.permute:
            self.permutations_list = []
            self.permutation_mapping = {}  ### Remove this when the mapping for permutation is corrected
            self.current_permutation = None
        
        if self.tile:
            self.tiling_combinations = []
            self.current_tiling_combination = None
        
        if self.unroll:
            self.unrolling_combinations = []
            self.current_unroll_combination = None

    def create_docker_commands(self):
        docker_base_command = "docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda soda-opt "
        soda_opt_bambu_pipeline = [
            "-affine-scalrep",
            "-cse",
            "-affine-data-copy-generate='generate-dma=false fast-mem-space=0'",
            "-erase-buffer-deallocation",
            "-promote-buffers-to-stack='max-rank-of-allocated-memref=4 max-alloc-size-in-bytes=4096'",
            "-lower-affine",
            "-convert-scf-to-cf",
            "-convert-memref-to-llvm",
            "-convert-math-to-llvm",
            "-convert-math-to-libm",
            "-arith-expand",
            "-memref-expand",
            "-convert-arith-to-llvm",
            "-convert-func-to-llvm='use-bare-ptr-memref-call-conv'",
            "-reconcile-unrealized-casts",
            "--mlir-print-ir-after-all",
            f"output/04b{self.current_configuration}.mlir",
            f"-o output/04c{self.current_configuration}.mlir",
            f"2>&1 | cat > output/05cintermediate-{self.current_configuration}.mlir"
        ]
        if self.permute:
            soda_opt_bambu_pipeline.insert(0, f"-test-loop-permutation='permutation-map={self.current_permutation}'")
            soda_command = docker_base_command + " ".join(soda_opt_bambu_pipeline)
        elif self.tile:
            tiling_combination_string = ",".join(str(i) for i in self.current_tiling_combination)
            if all(x == 0 for x in self.current_tiling_combination):
                soda_opt_bambu_pipeline.pop(4)
            else:
                soda_opt_bambu_pipeline.pop(4)
                soda_opt_bambu_pipeline.insert(0, f"-affine-loop-tile='tile-sizes={tiling_combination_string}'")
            soda_command = docker_base_command + " ".join(soda_opt_bambu_pipeline)
        elif self.unroll:
            soda_opt_bambu_pipeline.pop(2)
            soda_opt_bambu_pipeline.pop(2)
            loop_unroll_full_string = "-affine-loop-unroll='unroll-full'"
            loop_unroll_factor_string = f"-affine-loop-unroll='unroll-factor={self.current_unroll_combination[1]}'"
            if self.current_unroll_combination[1] == 0:
                loop_unroll_string_list = [loop_unroll_full_string] * self.current_unroll_combination[0]
            else:
                loop_unroll_string_list = [loop_unroll_full_string] * (self.current_unroll_combination[0] - 1) +  [loop_unroll_factor_string]
            loop_unroll_string = " ".join(loop_unroll_string_list)
            soda_opt_bambu_pipeline.insert(2, loop_unroll_string)
            soda_command = docker_base_command + " ".join(soda_opt_bambu_pipeline)
        print(f"SODA command: {soda_command}")
        self.commands = {
            "1a-soda": 
            f"docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda \
            soda-opt \
            -soda-outline-bambu-code \
            -soda-extract-arguments-to-xml='using-bare-ptr' \
            -soda-generate-bambu-accelcode \
            -convert-linalg-to-affine-loops \
            {self.layers[self.current_layer_name].file_path} \
            -o output/04a{self.current_configuration}.mlir \
            2>&1 | cat > output/05aintermediate-{self.current_configuration}.mlir",

            "1b-mlir": 
            f"docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda \
            mlir-opt \
            -expand-strided-metadata \
            output/04a{self.current_configuration}.mlir \
            -o output/04b{self.current_configuration}.mlir \
            2>&1 | cat > output/05bintermediate-{self.current_configuration}.mlir",

            "1c-soda": soda_command,

            "1d-mlir-opt": 
            f"docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda \
            mlir-opt \
            -symbol-dce \
            output/04c{self.current_configuration}.mlir \
            -o output/04d{self.current_configuration}.mlir \
            2>&1 | cat > output/05dintermediate-{self.current_configuration}.mlir",

            "1e-soda":
            f"docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda \
                mlir-translate -opaque-pointers=0  \
                --mlir-to-llvmir \
                output/04d{self.current_configuration}.mlir \
                -o output/05{self.current_configuration}.ll",

            "2-bambu":
            f"scripts/run-bambu.sh {self.current_configuration} 2>&1 | tee ./output/bambu-{self.current_configuration}.log",

            "3-openroad":
            f"scripts/run-openroad.sh {self.current_configuration} 2>&1 | tee ./output/openroad-{self.current_configuration}.log"
        }
        
    
    def create_or_append_to_csv(self, file_path, headers, data):
        """Create a CSV file with headers if it doesn't exist, or append data to it if it does."""
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers only if the file does not exist
            if not file_exists:
                writer.writerow(headers)
            
            # Append the data
            writer.writerow(data)


    def record_results(self, simulation_cycles, total_power, available_area, target_frequency = 100e6):
        
        giga_multiplier = 1e9
        current_layer = self.layers[self.current_layer_name]
        results_directory = "./results"
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        
        if self.current_layer_name.startswith("conv2d"):
            
            layer_info_header = ["configuration", "strides", "dilations",
                            "input_batch", "input_width", "input_height", "input_channel",
                            "kernel_width", "kernel_height", "kernel_input_channels", "kernel_output_channels",
                            "output_batch", "output_width", "output_height", "output_channel"]
            implemented_layer_info_header = ["actual_input_batch", "actual_input_width", "actual_input_height", "actual_input_channel",
                            "actual_kernel_width", "actual_kernel_height", "actual_kernel_input_channels", "actual_kernel_output_channels",
                            "actual_output_batch", "actual_output_width", "actual_output_height", "actual_output_channel",
                            "number_of_tiles"]
            results_header =  ["simulation_cycles", "total_power", "area", "runtime_in_s", "gflops", "gflops_per_watt", "energy_consumed", "flop_count"]
            
            if current_layer.clipped_output_channel is not None:
                output_channel = current_layer.clipped_output_channel
            else:
                output_channel = current_layer.output_channel
            if current_layer.clipped_input_channel is not None:
                input_channel = current_layer.clipped_input_channel
            else:
                input_channel = current_layer.input_channel
            
            actual_simulation_cycles = simulation_cycles * current_layer.no_of_tiles
            runtime_in_s = round(actual_simulation_cycles / target_frequency, 6)
            gflops = round(current_layer.flop_count / runtime_in_s / giga_multiplier, 6)
            gflops_per_watt = round(gflops / total_power, 6)
            energy_consumed = round(total_power * runtime_in_s, 12)


            layer_info = [self.current_configuration, current_layer.strides, current_layer.dilations,
                    current_layer.input_batch, current_layer.input_width, current_layer.input_height, current_layer.input_channel,
                    current_layer.kernel_width, current_layer.kernel_height, current_layer.kernel_input_channel, current_layer.kernel_output_channel, 
                    current_layer.output_batch, current_layer.output_width, current_layer.output_height, current_layer.output_channel]
            implemented_layer_info = [current_layer.input_batch, current_layer.input_width, current_layer.input_height, input_channel,
                    current_layer.kernel_width, current_layer.kernel_height, input_channel, output_channel, 
                    current_layer.output_batch, current_layer.output_width, current_layer.output_height, output_channel,
                    current_layer.no_of_tiles]
            
            results = {
                "Simulation Cycles": actual_simulation_cycles,
                "Total Power (W)": total_power,
                "Available Area (mm²)": available_area,
                "Runtime (s)": runtime_in_s,
                "GFLOPS": gflops,
                "GFLOPS/Watt": gflops_per_watt,
                "Energy Consumed (J)": energy_consumed,
                "FLOP Count": current_layer.flop_count
            }
            # Pretty print the results
            print("Simulation Results:")
            for key, value in results.items():
                print(f"{key}: {value}")
            
            if self.permute:
                file_path = f"./results/{self.model_name}_conv2d_permute.csv"
                permutation_order = list(map(int, self.current_permutation.split(',')))
                permutation_order_header = [f"permuation_order_{i}" for i in range(1, len(permutation_order)+1)]
                row_header = layer_info_header + permutation_order_header + implemented_layer_info_header + results_header
                row = layer_info + permutation_order + implemented_layer_info + list(results.values())
            elif self.tile:
                file_path = f"./results/{self.model_name}_conv2d_tile.csv"
                tiles = self.current_tiling_combination
                tiles_header = ["tiled_output_batch", "tiled_output_width", "tiled_output_height", "tiled_output_channel", "tiled_kernel_width", "tiled_kernel_height", "tiled_input_channel"]
                row_header = layer_info_header + tiles_header + implemented_layer_info_header + results_header
                row = layer_info + tiles + implemented_layer_info + list(results.values())
            elif self.unroll:
                file_path = f"./results/{self.model_name}_conv2d_unroll.csv"
                unroll_full, unrolling_factor = self.current_unroll_combination
                unrolls = [unroll_full, unrolling_factor]
                unroll_header = ["unroll_full", "unroll_factor"]
                row_header = layer_info_header + unroll_header + implemented_layer_info_header + results_header
                row = layer_info + unrolls + implemented_layer_info + list(results.values())


        elif self.current_layer_name.startswith("depthwise_conv2d"):
            
            layer_info_header = ["configuration", "strides", "dilation",
                            "input_batch", "input_width", "input_height", "input_channel",
                            "kernel_width", "kernel_height", "kernel_input_channel", "kernel_multiplier",
                            "output_batch", "output_width", "output_height", "output_channel", "output_multiplier"]
            implemented_layer_info_header = ["actual_input_batch", "actual_input_width", "actual_input_height", "actual_input_channel",
                            "actual_kernel_width", "actual_kernel_height", "actual_kernel_input_channels",
                            "actual_output_batch", "actual_output_width", "actual_output_height", "actual_output_channel",
                            "number_of_tiles"]
            results_header = ["simulation_cycles", "total_power", "area", "runtime_in_s", "gflops", "gflops_per_watt", "energy_consumed", "flop_count"]
            
            if current_layer.clipped_input_channel is not None:
                input_channel = current_layer.clipped_input_channel
            else:
                input_channel = current_layer.input_channel
            
            actual_simulation_cycles = simulation_cycles * current_layer.no_of_tiles
            runtime_in_s = round(actual_simulation_cycles / target_frequency, 6)
            gflops = round(current_layer.flop_count / runtime_in_s / giga_multiplier, 6)
            gflops_per_watt = round(gflops / total_power, 6)
            energy_consumed = round(total_power * runtime_in_s, 12)


            layer_info = [self.current_configuration, current_layer.strides, current_layer.dilations,
                    current_layer.input_batch, current_layer.input_width, current_layer.input_height, current_layer.input_channel,
                    current_layer.kernel_width, current_layer.kernel_height, current_layer.kernel_input_channel, current_layer.kernel_multiplier, 
                    current_layer.output_batch, current_layer.output_width, current_layer.output_height, current_layer.output_channel, current_layer.output_multiplier]
            implemented_layer_info = [current_layer.input_batch, current_layer.input_width, current_layer.input_height, input_channel,
                    current_layer.kernel_width, current_layer.kernel_height, input_channel,
                    current_layer.output_batch, current_layer.output_width, current_layer.output_height, input_channel,
                    current_layer.no_of_tiles]
            
            results = {
                "Simulation Cycles": actual_simulation_cycles,
                "Total Power (W)": total_power,
                "Available Area (mm²)": available_area,
                "Runtime (s)": runtime_in_s,
                "GFLOPS": gflops,
                "GFLOPS/Watt": gflops_per_watt,
                "Energy Consumed (J)": energy_consumed,
                "FLOP Count": current_layer.flop_count
            }
            # Pretty print the results
            print("Simulation Results:")
            for key, value in results.items():
                print(f"{key}: {value}")

            if self.permute:
                file_path = f"./results/{self.model_name}_depthwise_conv2d_permute.csv"
                permutation_order = list(map(int, self.current_permutation.split(',')))
                permutation_order_header = [f"permuation_order_{i}" for i in range(1, len(permutation_order)+1)]
                row_header = layer_info_header + permutation_order_header + implemented_layer_info_header + results_header
                row = layer_info + permutation_order + implemented_layer_info + list(results.values())
            elif self.tile:
                file_path = f"./results/{self.model_name}_depthwise_conv2d_tile.csv"
                tiles = self.current_tiling_combination
                tiles_header = ["tiled_output_batch", "tiled_output_width", "tiled_output_height", "tiled_input_channel", "tiled_kernel_width", "tiled_kernel_height"]
                row_header = layer_info_header + tiles_header + implemented_layer_info_header + results_header
                row = layer_info + tiles + implemented_layer_info + list(results.values())
            elif self.unroll:
                file_path = f"./results/{self.model_name}_depthwise_conv2d_unroll_1.csv"
                unroll_full, unrolling_factor = self.current_unroll_combination
                unrolls = [unroll_full, unrolling_factor]
                unroll_header = ["unroll_full", "unroll_factor"]
                row_header = layer_info_header + unroll_header + implemented_layer_info_header + results_header
                row = layer_info + unrolls + implemented_layer_info + list(results.values())

        elif self.current_layer_name.startswith("matmul"):
            layer_info_header = ["configuration",
                            "input_batch", "input_width", "input_height",
                            "weight_batch", "weight_width", "weight_height",
                            "output_batch", "output_width", "output_height"]
            implemented_layer_info_header = ["actual_input_batch", "actual_input_width", "actual_input_height",
                            "actual_weight_batch", "actual_weight_width", "actual_weight_height", 
                            "actual_output_batch", "actual_output_width", "actual_output_height",
                            "number_of_tiles"]
            results_header =  ["simulation_cycles", "total_power", "area", "runtime_in_s", "gflops", "gflops_per_watt", "energy_consumed", "flop_count"]
            
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
            if current_layer.clipped_input_width is not None:
                input_width = current_layer.clipped_input_width
            else:
                input_width = current_layer.input_width
            if current_layer.clipped_input_height is not None:
                input_height = current_layer.clipped_input_height
            else:
                input_height = current_layer.input_height
            
            actual_simulation_cycles = simulation_cycles * current_layer.no_of_tiles
            runtime_in_s = round(actual_simulation_cycles / target_frequency, 6)
            gflops = round(current_layer.flop_count / runtime_in_s / giga_multiplier, 6)
            gflops_per_watt = round(gflops / total_power, 6)
            energy_consumed = round(total_power * runtime_in_s, 12)


            layer_info = [self.current_configuration,
                    current_layer.input_batch, current_layer.input_width, current_layer.input_height,
                    current_layer.kernel_batch, current_layer.kernel_width, current_layer.kernel_height, 
                    current_layer.output_batch, current_layer.output_width, current_layer.output_height]  
            implemented_layer_info = [current_layer.input_batch, input_width, input_height,
                    current_layer.kernel_batch, kernel_width, kernel_height, 
                    current_layer.output_batch, output_width, output_height,
                    current_layer.no_of_tiles]
            
            results = {
                "Simulation Cycles": actual_simulation_cycles,
                "Total Power (W)": total_power,
                "Available Area (mm²)": available_area,
                "Runtime (s)": runtime_in_s,
                "GFLOPS": gflops,
                "GFLOPS/Watt": gflops_per_watt,
                "Energy Consumed (J)": energy_consumed,
                "FLOP Count": current_layer.flop_count
            }
            # Pretty print the results
            print("Simulation Results:")
            for key, value in results.items():
                print(f"{key}: {value}")

            if self.permute:
                file_path = f"./results/{self.model_name}_matmul_permute.csv"
                permutation_order = list(map(int, self.current_permutation.split(',')))
                permutation_order_header = [f"permuation_order_{i}" for i in range(1, len(permutation_order)+1)]
                row_header = layer_info_header + permutation_order_header + implemented_layer_info_header + results_header
                row = layer_info + permutation_order + implemented_layer_info + list(results.values())
            elif self.tile:
                file_path = f"./results/{self.model_name}_matmul_tile.csv"
                tiles = self.current_tiling_combination
                tiles_header = ["tiled_output_batch", "tiled_output_width", "tiled_output_height", "tiled_kernel_width"]
                row_header = layer_info_header + tiles_header + implemented_layer_info_header + results_header
                row = layer_info + tiles + implemented_layer_info + list(results.values())
            elif self.unroll:
                file_path = f"./results/{self.model_name}_matmul_unroll.csv"
                unroll_full, unrolling_factor = self.current_unroll_combination
                unrolls = [unroll_full, unrolling_factor]
                unroll_header = ["unroll_full", "unroll_factor"]
                row_header = layer_info_header + unroll_header + implemented_layer_info_header + results_header
                row = layer_info + unrolls + implemented_layer_info + list(results.values())
            
        self.create_or_append_to_csv(file_path, row_header, row)

    
    def execute_commands(self):
        # Define the paths
        txt_file_path = f"output/progress-{self.current_configuration}.txt"
        # Get the directory name
        directory = os.path.dirname(txt_file_path)
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Open the output text file in write mode
        with open(txt_file_path, "w") as output_file:

            # Initialize values for CSV
            simulation_cycles = None
            total_power = None
            available_area = None
            utilization_area = None

            for key, command in self.commands.items():
                # Execute the command
                subprocess.run(command, shell=True, stdout=output_file, stderr=output_file)

                # Check specific conditions after certain commands
                if key == "2-bambu":
                    cycles = ""
                    for runtime in open(f'output/{self.current_configuration}/bambu-log').readlines():
                        if "Average execution" in runtime:
                            cycles = [int(s) for s in runtime.split() if s.isdigit()][0]
                    
                    output_file.write("Average execution in cycles: {}\n".format(cycles))
                    simulation_cycles = int(cycles)  # Store value for CSV
                
                elif key == "3-openroad":
                    log_path_suffix = 'HLS_output/Synthesis/bash_flow/openroad/logs/nangate45/main_kernel/base/6_report.log'
                    gds_path_suffix = 'HLS_output/Synthesis/bash_flow/openroad/results/nangate45/main_kernel/base/6_final.gds'
                    log_file = f'output/{self.current_configuration}/' + log_path_suffix
                    power_multiplier = 1
                    for l in open(log_file, 'r').readlines():
                        if ("Total" in l and "Group" not in l):
                            total_power = float(l.split()[4]) * power_multiplier
                        if ("Design area" in l):
                            available_area = float(l.split()[2])
                            utilization_area = float(l.split()[4].strip('%'))
                    
                    output_file.write('Optimized accelerator:\n')
                    output_file.write('  total power consumption: {}W\n'.format(total_power))
                    output_file.write('  available chip area: {} um^2\n'.format(available_area))
                    output_file.write('  utilized chip area: {}%\n'.format(utilization_area))
                
                    self.record_results(simulation_cycles, total_power, available_area)

                    # Path to the output folder
                    output_folder = './output'
                    # Construct the command to find and delete files and folders
                    command = f"find {output_folder} -name '*{self.current_configuration}*' -exec rm -rf {{}} +"
                    # Execute the command
                    subprocess.run(command, shell=True, check=True)

                            
    def docker_commands(self):
        if self.permute:
            for permutation in self.permutations_list:
                self.current_configuration = f"{self.model_name}_permute_{self.current_layer_name}_{''.join(map(str, permutation))}"
                print("--------------------------------")
                print(f"Confgiuration: {self.current_configuration}")
                docker_perm_string = ','.join(map(str, permutation))  # String with commas for Docker command
                actual_perm_string = self.permutation_mapping.get(docker_perm_string, docker_perm_string)  # Retrieve actual perm string
                self.current_permutation = actual_perm_string
                print(f"Current permutation: {self.current_permutation}")
                self.create_docker_commands()
                self.execute_commands()
                self.current_permutation = None
            self.permutations_list = []
            self.permutation_mapping = {}
        elif self.tile:
            for count, tile in enumerate(self.tiling_combinations):
                self.current_configuration = f"{self.model_name}_tile_{self.current_layer_name}_{''.join(map(str, tile))}"
                print("--------------------------------")
                print(f"Confgiuration: {self.current_configuration}")
                self.current_tiling_combination = tile
                print(f"Current tiling combination: {self.current_tiling_combination}")
                self.create_docker_commands()
                self.execute_commands()
                self.current_tiling_combination = None
            self.tiling_combinations = []
        elif self.unroll:
            for count, unroll in enumerate(self.unrolling_combinations):
                self.current_configuration = f"{self.model_name}_unroll_{self.current_layer_name}_unroll_{unroll[0]}_factor_{unroll[1]}"
                print("--------------------------------")
                print(f"Confgiuration: {self.current_configuration}")
                self.current_unroll_combination = unroll
                print(f"Current unroll combination: {self.current_unroll_combination}")
                self.create_docker_commands()
                self.execute_commands()
                self.current_unroll_combination = None
            self.unrolling_combinations = []
  
    def get_permutations(self):
        if self.current_layer_name.startswith("conv2d"):
            mapping_csv_path = 'scripts/conv2d_mapping.csv'     ### Remove this when the mapping for permutation is corrected
            # group1 = [1, 2, 3]
            # group2 = [4, 6, 5]
            # #for perm1 in permutations(group1):
            # for perm1 in permutations(group1):
            #     permutation = [0] + list(perm1) + list(group2)
            #     self.permutations_list.append(permutation)
            # # for perm2 in permutations(group2):
            # #     for perm1 in permutations(group1):
            # #         permutation = [0] + list(perm2) + list(perm1)
            # #         self.permutations_list.append(permutation)
            self.permutations_list.append([0,1,2,3,4,5,6])
            self.permutations_list.append([0,2,1,3,5,6,4])
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            mapping_csv_path = 'scripts/depthwise_conv2d_mapping.csv'   ### Remove this when the mapping for permutation is corrected
            # Define the groups and generate all permutations
            group1 = [1, 2, 3]
            group2 = [4, 5]
            for perm1 in permutations(group1):
                for perm2 in permutations(group2):
                    permutation = [0] + list(perm1) + list(perm2)
                    self.permutations_list.append(permutation)
            group1 = [3, 4, 5]
            group2 = [1, 2]
            for perm1 in permutations(group1):
                for perm2 in permutations(group2):
                    permutation = [0] + list(perm1) + list(perm2)
                    self.permutations_list.append(permutation)
                    
        elif self.current_layer_name.startswith("matmul"):
            mapping_csv_path = 'scripts/matmul_mapping.csv' ### Remove this when the mapping for permutation is corrected
            group1 = [1, 2, 3]
            for perm in permutations(group1):
                permutation = [0] + list(perm)  # Add 0 at the beginning
                self.permutations_list.append(permutation)
        return mapping_csv_path ### Remove this when the mapping for permutation is corrected
        
    def perform_permutation(self):
        mapping_csv_path = self.get_permutations() ### Remove mapping_csv_path when the mapping for permutation is corrected
        ### Remove this when the mapping for permutation is corrected and use the permutations_list instead
        # Read the mapping CSV file into a dictionary
        with open(mapping_csv_path, mode='r', encoding='UTF-8') as mapping_file:
            reader = csv.reader(mapping_file)
            for row in reader:
                docker_perm = row[0]
                actual_perm = row[1]
                if actual_perm:  # Only add if the actual permutation is not empty
                    self.permutation_mapping[docker_perm] = actual_perm
        self.docker_commands()
     
    def generate_tiling_combinations(self, tiling_dimensions_1, tiling_dimensions_2, tiling_dimensions_3):
        current_layer = self.layers[self.current_layer_name]   
        if self.current_layer_name.startswith("conv2d"):
            output_tiles = tiling_dimensions_1
            kernel_tiles = tiling_dimensions_2
            input_channel_tiles = tiling_dimensions_3
            self.tiling_combinations.append([1, 14, 14, 1, 3, 3, 16])
            # for output in output_tiles:
            #     for kernel in kernel_tiles:
            #         if output>=kernel:
            #             for input_channel in input_channel_tiles:
            #                 tiling_combination_string = [current_layer.output_batch, output, output, current_layer.clipped_output_channel, 
            #                                              kernel, kernel, input_channel]
            #                 self.tiling_combinations.append(tiling_combination_string)
            # self.tiling_combinations.pop(-1)
            print(f"Tiling combinations: {self.tiling_combinations}")
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            output_tiles = tiling_dimensions_1
            kernel_tiles = tiling_dimensions_2
            input_channel_tiles = tiling_dimensions_3
            self.tiling_combinations.append([0, 0, 0, 0, 0, 0])
            for output in output_tiles:
                for kernel in kernel_tiles:
                    if output>=kernel:
                        for input_channel in input_channel_tiles:
                            tiling_combination = [current_layer.output_batch, output, output, input_channel,
                                                  kernel, kernel]
                            self.tiling_combinations.append(tiling_combination)
            self.tiling_combinations.pop(-1)
            print(f"Tiling combinations: {self.tiling_combinations}")
        elif self.current_layer_name.startswith("matmul"):
            output_width_tiles = tiling_dimensions_1
            output_height_tiles = tiling_dimensions_2
            kernel_width_tiles = tiling_dimensions_3
            self.tiling_combinations.append([1, 1, 64, 16])          
            # for output_width in output_width_tiles:
            #     for output_height in output_height_tiles:
            #         for kernel_width in kernel_width_tiles:
            #             tiling_combination = [current_layer.output_batch, output_width, output_height, kernel_width]
            #             self.tiling_combinations.append(tiling_combination)
            # self.tiling_combinations.pop(-1)
            print(f"Tiling combinations: {self.tiling_combinations}")
    
    def get_tile_sizes(self, max_tile_size, is_kernel = False, min_power = 2):
        if is_kernel:
            kernel_tile_sizes = []
            kernel_tile_sizes = [i for i in range(3, max_tile_size+1, 2)]
            if max_tile_size not in kernel_tile_sizes:
                kernel_tile_sizes.append(max_tile_size)
            return kernel_tile_sizes
        else:
            power_of_two = [2**i for i in range(min_power, int(np.log2(max_tile_size))+1)]
            output_tile_sizes = []
            output_tile_sizes = [i for i in power_of_two if i < max_tile_size]
            if max_tile_size not in output_tile_sizes:
                output_tile_sizes.append(max_tile_size)
            return output_tile_sizes
            
    
    def get_tiling_combinations(self):
        
        current_layer = self.layers[self.current_layer_name]

        if self.current_layer_name.startswith("conv2d"):
            output_tiles = self.get_tile_sizes(current_layer.output_height)
            kernel_tiles = self.get_tile_sizes(current_layer.kernel_height, is_kernel=True)
            if current_layer.clipped_input_channel is not None:
                input_channel_tiles = self.get_tile_sizes(current_layer.clipped_input_channel)
            else:
                input_channel_tiles = self.get_tile_sizes(current_layer.input_channel)
            self.generate_tiling_combinations(output_tiles, kernel_tiles, input_channel_tiles)  
        
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            output_tiles = self.get_tile_sizes(current_layer.output_height)
            kernel_tiles = self.get_tile_sizes(current_layer.kernel_height, is_kernel=True)
            if current_layer.clipped_input_channel is not None:
                input_channel_tiles = self.get_tile_sizes(current_layer.clipped_input_channel)
            else:
                input_channel_tiles = self.get_tile_sizes(current_layer.input_channel)
            self.generate_tiling_combinations(output_tiles, kernel_tiles, input_channel_tiles)
        
        elif self.current_layer_name.startswith("matmul"):
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
            output_width_tiles = self.get_tile_sizes(output_width)
            output_height_tiles = self.get_tile_sizes(output_height)
            kernel_width_tiles = self.get_tile_sizes(kernel_width)
            self.generate_tiling_combinations(output_width_tiles, output_height_tiles, kernel_width_tiles)
    
    def perform_tiling(self):
        self.get_tiling_combinations()
        self.docker_commands()

    
    def generate_unrolling_combinations(self, unrolls_1, unrolls_2, unrolls_3 = None):
        self.unrolling_combinations.append((0,0))
        # For unroll-1
        for i in unrolls_1:
            self.unrolling_combinations.append((1,i))
        self.unrolling_combinations.append((1,0))
        # For unroll-2
        for i in unrolls_2:
            self.unrolling_combinations.append((2,i))
        self.unrolling_combinations.append((2,0))
        # For unroll-3
        if unrolls_3 is not None:
            for i in unrolls_3:
                self.unrolling_combinations.append((3,i))
            self.unrolling_combinations.append((3,0))
        print(f"Unrolling combinations: {self.unrolling_combinations}")
   
    def get_unroll_sizes(self, max_unroll_size, min_power = 2):
        powers_of_two = [2**i for i in range(min_power, int(np.log2(max_unroll_size)) + 1)]
        return [p for p in powers_of_two if p < max_unroll_size]
       
    def get_unrolling_combinations(self):
        current_layer = self.layers[self.current_layer_name]
        if self.current_layer_name.startswith("conv2d"):
            kernel_unrolls = self.get_unroll_sizes(current_layer.kernel_height)
            if current_layer.clipped_input_channel is not None:
                input_channel_unrolls = self.get_unroll_sizes(current_layer.clipped_input_channel)
            else:
                input_channel_unrolls = self.get_unroll_sizes(current_layer.input_channel)
            #self.generate_unrolling_combinations(input_channel_unrolls, kernel_unrolls, kernel_unrolls)  
            self.unrolling_combinations.append((0,0))
            self.unrolling_combinations.append((3,0))
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            kernel_unrolls = self.get_unroll_sizes(current_layer.kernel_height)
            if current_layer.clipped_input_channel is not None:
                input_channel_unrolls = self.get_unroll_sizes(current_layer.clipped_input_channel)
            else:
                input_channel_unrolls = self.get_unroll_sizes(current_layer.input_channel)
            #self.generate_unrolling_combinations(kernel_unrolls, kernel_unrolls, input_channel_unrolls)
            #self.unrolling_combinations.append((0,0))
            self.unrolling_combinations.append((3,0))
        elif self.current_layer_name.startswith("matmul"):
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
            #output_width_unrolls = self.get_unroll_sizes(output_width)
            output_height_unrolls = self.get_unroll_sizes(output_height)
            kernel_width_unrolls = self.get_unroll_sizes(kernel_width)
            #self.generate_unrolling_combinations(kernel_width_unrolls, output_height_unrolls)
            self.unrolling_combinations.append((0,0))
            self.unrolling_combinations.append((2,0))
 
    def perform_unrolling(self):
        self.get_unrolling_combinations()
        self.docker_commands()


    def execute(self):
        if self.permute:
            for layer_name in self.layers.keys():
                print("===========================")
                print(f"Layer name: {layer_name}")
                self.current_layer_name = layer_name
                self.perform_permutation()
        elif self.tile:
            for layer_name in self.layers.keys():
                print("===========================")
                print(f"Layer name: {layer_name}")
                self.current_layer_name = layer_name
                self.perform_tiling()
        elif self.unroll:
            for layer_name in self.layers.keys():
                print("===========================")
                print(f"Layer name: {layer_name}")
                self.current_layer_name = layer_name
                self.perform_unrolling()