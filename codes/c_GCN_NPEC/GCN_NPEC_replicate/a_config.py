import pickle
import os
import argparse
from datetime import datetime

def arg_parser():
	parser = argparse.ArgumentParser()
	# model param
	parser.add_argument('-m', '--mode', metavar = 'M', type = str, default = 'train', choices = ['train', 'test'], help = 'train or test')
	parser.add_argument('-n', '--n_customer', metavar = 'N', type = int, default = 20, help = 'number of customer nodes, time sequence')
	parser.add_argument("-hm", "--hidden_dim", metavar='HM', type=int, default=100, help="hidden dimension")
	parser.add_argument("-gn", "--gcn_num_layers", metavar='GN', type=int, default=3, help="number of GCN layers")
	parser.add_argument("-k", "--k", metavar='K', type=int, default=10, help="k-nearest neighbor")
	parser.add_argument("-nd", "--node_info_dim", metavar='ND', type=int, default=1, help="node information dimension")
	parser.add_argument("-grn", "--gru_num_layers", metavar='GRN', type=int, default=2, help="number of GRU layers")
	parser.add_argument("-mln", "--mlp_num_layers", metavar='MLN', type=int, default=2, help="number of MLP layers")
	parser.add_argument("-tp", "--time_penalty", metavar='TP', type=float, default=0.001, help="time penalty")
 
	# pretrain param
	parser.add_argument("-pe", "--pretrain_epoch", metavar='PE', type=int, default=0, help="number of pretrain epoch")
	parser.add_argument("-plr", "--pretrain_lr", metavar='PLR', type=float, default=1e-3, help="pretrain learning rate")

	# train param
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 64, help = 'batch size')
	parser.add_argument('-bs', '--batch_steps', metavar="BS", type=int, default=1000, help="number of batch steps")
	parser.add_argument('-e', '--epochs', metavar = 'E', type = int, default = 50, help = 'total number of samples = epochs * number of samples')
	parser.add_argument('-lr', '--lr', metavar = 'LR', type = float, default = 1e-3, help = 'initial learning rate')
	parser.add_argument('-alpha', '--alpha', metavar='ALPHA', type=float, default=1.0, help="loss weight for REINFORCE")
	parser.add_argument('-beta', '--beta', metavar='BETA', type=float, default=1.0, help="loss weight for SUPERVISE")

	# store param
	time_stamp = datetime.now().strftime('%m%d_%H_%M')
	rd = "../new_record/" + time_stamp + "/result/"
	wd = "../new_record/" + time_stamp + "/weight/"
	pd = "../new_record/" + time_stamp + "/pkl/"
	parser.add_argument('-dd', '--data_dir', metavar = 'DD', type = str, default = '../VRP20_data/', help = 'data dir')
	parser.add_argument('-rd', '--result_dir', metavar = 'RD', type = str, default = rd, help = 'result dir')
	parser.add_argument('-wd', '--weight_dir', metavar = 'MD', type = str, default = wd, help = 'model weight save dir')
	parser.add_argument('-pd', '--pkl_dir', metavar = 'PD', type = str, default = pd, help = 'pkl save dir')
	parser.add_argument("-desc", "--description", metavar = "DESC", type = str, default = "None", help = "description")
	args = parser.parse_args()
	return args

class Config():
	def __init__(self, **kwargs):	
		for k, v in kwargs.items():
			self.__dict__[k] = v
		self.task = 'VRP%d_%s'%(self.n_customer, self.mode)
		self.dump_date = datetime.now().strftime('%m%d_%H_%M')
		for x in [self.data_dir, self.result_dir, self.weight_dir, self.pkl_dir]:
			os.makedirs(x, exist_ok = True)
		self.pkl_path = self.pkl_dir + self.task + '.pkl'

def dump_pkl(args, verbose = True, param_log = True):
	cfg = Config(**vars(args))
	with open(cfg.pkl_path, 'wb') as f:
		pickle.dump(cfg, f)
		print('--- save pickle file in %s ---\n'%cfg.pkl_path)
		if verbose:
			print(''.join('%s: %s\n'%item for item in vars(cfg).items()))
		if param_log:
			path = '%sparam_%s_%s.csv'%(cfg.result_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
			with open(path, 'w') as f:
				f.write(''.join('%s,%s\n'%item for item in vars(cfg).items())) 
	
def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')
	with open(pkl_path, 'rb') as f:
		cfg = pickle.load(f)
		if verbose:
			print(''.join('%s: %s\n'%item for item in vars(cfg).items()))
	return cfg

def train_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, 
						default = 'pkl/VRP20_train.pkl',
						help = 'pkl/VRP***_train.pkl, pkl file only, default: pkl/VRP20_train.pkl')
	parser.add_argument("-dp", "--data_path", metavar='DP', type=str, default='', help="data path")
	parser.add_argument("-tdp", "--test_data_path", metavar='TDP', type=str, default='', help="test data path")
	parser.add_argument("-pdp", "--plot_data_path", metavar="PDP", type=str, default='', help="plot data path")
	parser.add_argument("-mp", "--model_path", metavar="MP", type=str, default = "", help="continue with the existing model")
	parser.add_argument("-mpcd", "--model_path_classification_decoder", metavar="MPCD", type=str, default = "", help="continue with the pretrained classification decoder")
	parser.add_argument("-mpgcn", "--model_path_gcn", metavar="MPGCN", type=str, default = "", help="continue with the pretrained gcn")
	parser.add_argument("-mpsd", "--model_path_sequential_decoder", metavar="MPSD", type=str, default = "", help="continue with the pretrained sequential decoder")
	args = parser.parse_args()
	return args

def data_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, 
					default = 'pkl/VRP20_train.pkl',
					help = 'pkl/VRP***_train.pkl, pkl file only, default: pkl/VRP20_train.pkl')
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = arg_parser()
	dump_pkl(args)