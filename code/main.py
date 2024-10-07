from pathlib import Path
from parse_arguments import parse_arguments
from read_mlir import read_file
from create_mlir_files import MlirFiles
from design_space_exploration import DSE

def main():
    args = parse_arguments()
    print(args)
    if args.read_mlir is not None:
        path = Path(args.read_mlir)
        if path.exists():
            layers = read_file(args)
            if args.start_layer is not None:
                start_layer = int(args.start_layer)
                if args.end_layer is not None:
                    end_layer = int(args.end_layer)
                else:
                    end_layer = len(layers)
                if start_layer > 0 and end_layer <= len(layers):
                    # Iterating through the original dictionary
                    filtered_layers = {}
                    for key, value in layers.items():
                        number = int(key.split("_")[-1])
                        # Check if the number is greater than 2
                        if number >= start_layer and number <= end_layer:
                            filtered_layers[key] = value
                    layers = filtered_layers
                else:
                    print("Start layer out of range!")
            elif args.select_layer is not None:
                select_layer = int(args.select_layer)
                if select_layer > 0 and select_layer <= len(layers):
                    # Iterating through the original dictionary
                    filtered_layers = {}
                    for key, value in layers.items():
                        number = int(key.split("_")[-1])
                        # Check if the number is greater than 2
                        if number == select_layer:
                            filtered_layers[key] = value
                            break
                    layers = filtered_layers
                else:
                    print("Select layer out of range!")
            layers = MlirFiles(args, layers).execute()
            layers = DSE(args, layers).execute()
        else:
            print("MLIR file doesn't exist!")
    else:
        print("No input MLIR file given!")

if __name__ == "__main__":
    main()
