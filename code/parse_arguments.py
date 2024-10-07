import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_mlir", action = "store", help = "Reads .mlir files.")
    parser.add_argument("--permute", action = "store_true", help = "Explores loop permutation.")
    parser.add_argument("--tile", action = "store_true", help = "Explores loop tiling.")
    parser.add_argument("--unroll", action = "store_true", help = "Explores loop unrolling.")
    parser.add_argument("--conv2d", action = "store_true", help = "Explores 2D Convolution layers.")
    parser.add_argument("--depthwise_conv2d", action = "store_true", help = "Explores Depthwise 2D Convolution layers.")
    parser.add_argument("--matmul", action = "store_true", help = "Explores Fully Connected layers.")
    parser.add_argument("--start_layer", action = "store", help = "Which layer to start exploration from.")
    parser.add_argument("--end_layer", action = "store", help = "Which layer to end exploration at.")
    parser.add_argument("--select_layer", action = "store", help = "Only the selected layer is explored.")
    args = parser.parse_args()
    return args
