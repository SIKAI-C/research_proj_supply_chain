import os
import argparse
import torch

from .a_config import load_pkl, Config
from .d_model import Model

def convertor_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", metavar="M", type=str, default="aggregate", choices=["split", "aggregate"], help="split the whole model to separeted parts or the reversed process")
    parser.add_argument("-rd", "--result_dir", metavar="RD", type=str, default="./result/", help="the direction used for storing the ")
    parser.add_argument("-pp", "--pkl_path", metavar="PP", type=str, default="", help="the path of pkl")
    parser.add_argument("-mp", "--model_path", metavar="MP", type=str, default="", help="the path of the whole model")
    parser.add_argument("-mpcd", "--model_path_classification_decoder", metavar="MPCD", type=str, default="", help="the path of the classification decoder")
    parser.add_argument("-mpgcn", "--model_path_gcn", metavar="MPGCN", type=str, default="", help="the path of the GCN")
    parser.add_argument("-mpsd", "--model_path_sequential_decoder", metavar="MPSD", type=str, default="", help="the path of the sequential decoder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = convertor_parser()
    cfg = load_pkl(args.pkl_path)
    os.makedirs(args.result_dir, exist_ok=True)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    model = Model(
        hidden_dim=cfg.hidden_dim,
        gcn_num_layers=cfg.gcn_num_layers,
        k=cfg.k,
        node_info_dim=cfg.node_info_dim,
        gru_num_layers=cfg.gru_num_layers,
        mlp_num_layers=cfg.mlp_num_layers
    ).to(device)
    
    mode = args.mode
    if mode == "split":
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        torch.save(model.GCNEncoder.state_dict(), args.result_dir+"gcn.pth")
        torch.save(model.sequentialDecoderSample.state_dict(), args.result_dir+"sequential_decoder_sample.pth")
        torch.save(model.sequentialDecoderGreedy.state_dict(), args.result_dir+"sequential_decoder_greedy.pth")
        torch.save(model.classificationDecoder.state_dict(), args.result_dir+"classification_decoder.pth")
    elif mode == "aggregate":
        model.GCNEncoder.load_state_dict(torch.load(args.model_path_gcn, map_location=device))
        model.sequentialDecoderSample.load_state_dict(torch.load(args.model_path_sequential_decoder, map_location=device))
        model.sequentialDecoderGreedy.load_state_dict(torch.load(args.model_path_sequential_decoder, map_location=device))
        model.classificationDecoder.load_state_dict(torch.load(args.model_path_classification_decoder, map_location=device))
        torch.save(model.state_dict(), args.result_dir+"model.pth")
    else:
        print("Error! Invalid Keyword for -m (mode)")
        