import os
import argparse
import numpy as np

from .a_config import load_pkl, Config

def merge_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d1", "--data1", metavar="D1", type=str, default="./data1.npz", help="the path of the first data")
    parser.add_argument("-d2", "--data2", metavar="D2", type=str, default="./data2.npz", help="the path of the second data")
    parser.add_argument("-o", "--output", metavar="O", type=str, default="./data.npz", help="the path of the output data")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = merge_parser()
    data1_path = args.data1
    data2_path = args.data2
    data1 = np.load(data1_path)
    data2 = np.load(data2_path)
    com_data = {}
    print(data1)
    for key in data1.files:
        com_data[key] = np.concatenate((data1[key], data2[key]))
    store_path = args.output
    np.savez(store_path, **com_data)
    