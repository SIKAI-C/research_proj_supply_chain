# RL folder

## Outline
- ./agentHOF/: agent_Hall_of_Fame, record the best agent in each setting
- ./GCN_NPEC_replicate/: GCN_NPEC model
- ./record/: record the training/testing process

## ./agentHOF/
- store the .pkl file with the .pth file
- store the model.pth which contains the whole model, or store each part of the model separately, like, gcn.pth, classification_decoder.pth, etc.

## ./GCN_NPEC_replicate/
- the most comprehensive model so far
- a_config.py
  - `python3 a_config.py -n 5 -b 256 -bs 195 -e 1000 -hm 256 -lr 0.001 -k 5`, set the hyperparameters
  - this command will mkdir folders "RL/record/yyyy_mm_dd/pkl(result, weight)" and store the hyperparameters in the file:
    - "RL/record/yyyy_mm_dd/pkl/VRPxx_train.pkl"
    - "RL/record/yyyy_mm_dd/result/params_VRPxx_train_mmdd_hh_mm.csv"
- a_utilities.py
- b_attention.py
- b_context.py
- c_decoder.py
- c_GCN.py
- d_beam_search.py
- d_env.py
- d_model.py
- e_dataset.py
- e_generate_instance.py
  - `python3 e_generate_instance.py -p RL/record/yyyy_mm_dd/pkl/VRPxx_train.pkl`
  - generate the real/synthetic data according to the .pkl file, batch_size * batch_step
- f_pretrain.py
- f_train_test.py
- g_main.py
  - `python3 g_main.py -p RL/record/yyyy_mm_dd/pkl/VRPxx_train.pkl -dp training_data_path -tdp testing_data_path (-mp model_path -mpcd model_path_classification_decoder -mpsd model_path_sequential decoder -mpgcn model_path_gcn)`
  - training the model from the initial state or continue training from the -mp(mpgcn, mpcd, mpsd)
- g_supervised_without_sd.py
  - like g_supervised.py, but without the sequential_decoder
- g_supervised.py
  - `python3 g_supervised.py -p RL/record/yyyy_mm_dd/pkl/VRPxx_train.pkl -dp training_data_path -tdp testing_data_path (-mp model_path -mpcd model_path_classification_decoder -mpsd model_path_sequential decoder -mpgcn model_path_gcn)`
  - pretrain the model by using the supervise framework - use the optimal solution as target (just like the nlp model, compute the loss for each step)
- h_agent_convertor.py
  - in the pretrain phase, we store each part separately. in the formal training phase, we train the whole model. so we need to convert the model from one form to another.
  - aggregate the gcn, classification_decoder, sequential_decoder into one model `python3 h_agent_convertor -m aggregate -rd result_dir -pp pkl_path -mpcd xx -mpgcn xx -mpsd xx`
  - split the model to gcn, classification_decoder, sequential_decoder `python3 h_agent_convertor -m split -rd result_dir -mp model_path`
- h_learn_curve_so_far.py
  - plot the learning curve of the model
  - `python3 h_learn_curve_so_far.py -tp .txt_path -ip image_dir`
- h_merge_two_npz.py
  - merge two data files generated at different time
  - `python3 h_merge_two_npz.py -d1 data_file1 -d2 data_file2 -o output_file`

## ./record/ 
- ./record/yyyy_mm_dd/
  - image/ - image for learning curve
    - TEST_d_vs_epoch.png
    - TEST_dist_vs_epoch.png
    - TEST_loss_vs_epoch.png
    - Train_d_vs_epoch.png
    - Train_d_vs_step.png
    - Train_dist_vs_epoch.png
    - Train_dist_vs_step.png
    - Train_is_update_vs_epoch.png
    - Train_loss_vs_epoch.png
    - Train_loss_vs_step.png
  - pkl/
    - VRPxx_train.pkl
  - result/
    - image/ - image for classification decoder
      - e_xx_b_xx.png
    - pretrain/
      - VRPxx_mmdd_hh_mm.txt
    - params_VRPxx_train_mmdd_hh_mm.csv
    - training_VRPxx_mmdd_hh_mm.txt
  - weight/
    - pretrain/
      - checkpoints_epoch_xx.pth
    - checkpoints_epoch_xx.pth
