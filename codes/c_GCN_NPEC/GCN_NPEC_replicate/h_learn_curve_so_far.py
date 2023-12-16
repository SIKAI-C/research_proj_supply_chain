import os
import argparse
import matplotlib.pyplot as plt

from .a_config import load_pkl, Config
from .d_model import Model

def plot_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tp", "--txt_path", metavar="TD", type=str, default="./training.txt", help="the path of the record")
    parser.add_argument("-ip", "--img_path", metavar="ID", type=str, default="./learning_curve_so_far.png", help="the path of the image")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = plot_parser()
    os.makedirs(args.img_path, exist_ok=True)
    
    txt_path = args.txt_path
    
    tr_s_str = ["s_dist", "g_dist", "b_dist", "l", "l_r", "l_s", "d", "b_d"]
    tr_e_str = ["s_dist", "g_dist", "b_dist", "l", "l_r", "l_s", "d", "b_d", "is_update"]
    te_e_str = ["s_dist", "g_dist", "b_dist", "l", "l_r", "l_s", "d", "b_d"]
    
    tr_s = {}
    tr_e = {}
    te_e = {}
    
    for s in tr_s_str: tr_s[s] = []
    for s in tr_e_str: tr_e[s] = []
    for s in te_e_str: te_e[s] = []
    
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("e"):
                line = line.split("|")
                tr_s["s_dist"].append(float(line[2].split()[-1]))
                tr_s["g_dist"].append(float(line[3].split()[-1]))
                tr_s["b_dist"].append(float(line[4].split()[-1]))
                tr_s["l"].append(float(line[5].split()[-1]))
                tr_s["l_r"].append(float(line[6].split()[-1]))
                tr_s["l_s"].append(float(line[7].split()[-1]))
                tr_s["d"].append(float(line[8].split()[-1]))
                tr_s["b_d"].append(float(line[9].split()[-1]))
            elif line.endswith("True\n") or line.endswith("False\n"):
                line = line.split("|")
                tr_e["s_dist"].append(float(line[0].split()[-1]))
                tr_e["g_dist"].append(float(line[1].split()[-1]))
                tr_e["b_dist"].append(float(line[2].split()[-1]))
                tr_e["l"].append(float(line[3].split()[-1]))
                tr_e["l_r"].append(float(line[4].split()[-1]))
                tr_e["l_s"].append(float(line[5].split()[-1]))
                tr_e["d"].append(float(line[6].split()[-1]))
                tr_e["b_d"].append(float(line[7].split()[-1]))
                tr_e["is_update"].append(line[8].split()[-1] == "True")
            else:
                line = line.split("|")
                te_e["g_dist"].append(float(line[1].split()[-1]))
                te_e["s_dist"].append(float(line[0].split()[-1]))
                te_e["b_dist"].append(float(line[2].split()[-1]))
                te_e["l"].append(float(line[3].split()[-1]))
                te_e["l_r"].append(float(line[4].split()[-1]))
                te_e["l_s"].append(float(line[5].split()[-1]))
                te_e["d"].append(float(line[6].split()[-1]))
                te_e["b_d"].append(float(line[7].split()[-1]))
    
    # train-step
    if tr_s:
        img_title = "TRAIN_dist_vs_step"
        img_path = args.img_path + img_title + ".png"
        plt.plot(tr_s["s_dist"], label="s_dist")
        plt.plot(tr_s["g_dist"], label="g_dist")
        plt.plot(tr_s["b_dist"], label="b_dist")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
        
        img_title = "TRAIN_loss_vs_step"
        img_path = args.img_path + img_title + ".png"
        plt.plot(tr_s["l"], label="l")
        plt.plot(tr_s["l_r"], label="l_r")
        plt.plot(tr_s["l_s"], label="l_s")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
        
        img_title = "TRAIN_d_vs_step"
        img_path = args.img_path + img_title + ".png"
        plt.plot(tr_s["d"], label="d")
        plt.plot(tr_s["b_d"], label="b_d")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
    
    # train-epoch
    if tr_e:
        img_title = "TRAIN_dist_vs_epoch"
        img_path = args.img_path + img_title + ".png"
        plt.plot(tr_e["s_dist"], label="s_dist")
        plt.plot(tr_e["g_dist"], label="g_dist")
        plt.plot(tr_e["b_dist"], label="b_dist")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
        
        img_title = "TRAIN_loss_vs_epoch"
        img_path = args.img_path + img_title + ".png"
        plt.plot(tr_e["l"], label="l")
        plt.plot(tr_e["l_r"], label="l_r")
        plt.plot(tr_e["l_s"], label="l_s")
        plt.legend()
        plt.title(img_title) 
        plt.savefig(img_path)
        plt.clf()
        
        img_title = "TRAIN_d_vs_epoch"
        img_path = args.img_path + img_title + ".png"
        plt.plot(tr_e["d"], label="d")
        plt.plot(tr_e["b_d"], label="b_d")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
        
        img_title = "TRAIN_is_update_vs_epoch"
        img_path = args.img_path + img_title + ".png"
        plt.plot(tr_e["is_update"], label="is_update")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
        
    # test-epoch
    if te_e:
        img_title = "TEST_dist_vs_epoch"
        img_path = args.img_path + img_title + ".png"
        plt.plot(te_e["s_dist"], label="s_dist")
        plt.plot(te_e["g_dist"], label="g_dist")
        plt.plot(te_e["b_dist"], label="b_dist")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
        
        img_title = "TEST_loss_vs_epoch"
        img_path = args.img_path + img_title + ".png"
        plt.plot(te_e["l"], label="l")
        plt.plot(te_e["l_r"], label="l_r")
        plt.plot(te_e["l_s"], label="l_s")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
        
        img_title = "TEST_d_vs_epoch"
        img_path = args.img_path + img_title + ".png"
        plt.plot(te_e["d"], label="d")
        plt.plot(te_e["b_d"], label="b_d")
        plt.legend()
        plt.title(img_title)
        plt.savefig(img_path)
        plt.clf()
    

