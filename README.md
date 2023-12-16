
```
research_proj_supply_chain
├─ .DS_Store
├─ README.md
├─ codes
│  ├─ .DS_Store
│  ├─ __init__.py
│  ├─ a_CVRP
│  │  ├─ __init__.py
│  │  ├─ a_tabu_search
│  │  │  ├─ __init__.py
│  │  │  ├─ a_node_aggregation.py
│  │  │  ├─ b_fenge_input.py
│  │  │  ├─ c_evaluate_neighborhood.py
│  │  │  ├─ c_get_neighbour.py
│  │  │  ├─ c_initial_solution.py
│  │  │  ├─ c_parser_solution.py
│  │  │  ├─ c_report.py
│  │  │  ├─ d_summary.py
│  │  │  ├─ d_tabu_framework.py
│  │  │  ├─ e_experiment.py
│  │  │  └─ experiment.ipynb
│  │  └─ b_mix_integer_programming
│  │     ├─ __init__.py
│  │     ├─ a_mip_copt_solution.py
│  │     ├─ a_mip_solution.py
│  │     ├─ b_mip_experiment.py
│  │     └─ experiment.ipynb
│  ├─ b_IRP
│  │  ├─ __init__.py
│  │  ├─ a_lagrangian_nultiplier
│  │  │  ├─ a_generate_instance.py
│  │  │  ├─ a_gurobi_solution_parser.py
│  │  │  └─ b_lagrangian_multipliers_1.py
│  │  └─ b_fixed_partition_policy
│  │     ├─ a_generate_instance.py
│  │     ├─ b_rFPP.py
│  │     ├─ c_solution_parser.py
│  │     └─ test_FPP.ipynb
│  └─ c_GCN_NPEC
│     ├─ .DS_Store
│     ├─ GCN_NPEC_replicate
│     │  ├─ a_config.py
│     │  ├─ a_utilities.py
│     │  ├─ b_attention.py
│     │  ├─ b_context.py
│     │  ├─ c_GCN.py
│     │  ├─ c_decoder.py
│     │  ├─ d_beam_search.py
│     │  ├─ d_env.py
│     │  ├─ d_model.py
│     │  ├─ e_dataset.py
│     │  ├─ e_generate_instance.py
│     │  ├─ f_pretrain.py
│     │  ├─ f_train_test.py
│     │  ├─ g_main.py
│     │  ├─ g_supervised.py
│     │  ├─ g_supervised_withoud_sd.py
│     │  ├─ h_agent_convertor.py
│     │  ├─ h_learn_curve_so_far.py
│     │  └─ h_merge_two_npz.py
│     ├─ __init__.py
│     ├─ agentHOF
│     │  ├─ VRP20_best0
│     │  │  ├─ classification_decoder.pth
│     │  │  ├─ gcn.pth
│     │  │  ├─ model.pkl
│     │  │  ├─ model.pth
│     │  │  ├─ sequential_decoder_greedy.pth
│     │  │  └─ sequential_decoder_sample.pth
│     │  ├─ VRP20_best1
│     │  │  ├─ model.pkl
│     │  │  └─ model.pth
│     │  ├─ VRP5_best0
│     │  │  ├─ model.pkl
│     │  │  └─ model.pth
│     │  └─ VRP5_pretrain0
│     │     ├─ VRP5_train.pkl
│     │     ├─ checkpoints_epoch_51_GCN.pth
│     │     ├─ checkpoints_epoch_51_classification_decoder.pth
│     │     ├─ checkpoints_epoch_51_sequential_decoder.pth
│     │     └─ model.pth
│     ├─ instruction.md
│     └─ record
│        ├─ 0721_18_33
│        │  ├─ pkl
│        │  │  └─ VRP20_train.pkl
│        │  └─ result
│        │     ├─ param_VRP20_train_0721_18_33.csv
│        │     └─ pretrain
│        │        └─ VRP20_0721_18_34.txt
│        ├─ 0721_18_44
│        │  ├─ pkl
│        │  │  └─ VRP20_train.pkl
│        │  ├─ result
│        │  │  ├─ param_VRP20_train_0721_18_44.csv
│        │  │  └─ pretrain
│        │  │     └─ VRP20_0721_18_45.txt
│        │  └─ weight
│        │     └─ pretrain
│        │        ├─ checkpoints_epoch_96_GCN.pth
│        │        ├─ checkpoints_epoch_96_classification_decoder.pth
│        │        └─ checkpoints_epoch_96_sequential_decoder.pth
│        └─ 0724_09_55
│           ├─ pkl
│           │  └─ VRP20_train.pkl
│           ├─ result
│           │  ├─ param_VRP20_train_0724_09_55.csv
│           │  └─ training_VRP20_0724_09_59.txt
│           └─ weight
│              └─ checkpoints_epoch_100.pth
└─ docs
   ├─ .DS_Store
   ├─ DIMACS-IRP
   │  ├─ A Three-Stage Matheuristic for Multi-vehicle Inventory Routing Problem.pdf
   │  └─ IRP-instances-description.pdf
   ├─ EURO-VRP
   │  ├─ rank10_paper.pdf
   │  ├─ rank10_slides.pdf
   │  ├─ rank1_paper.pdf
   │  ├─ rank1_slides.pdf
   │  ├─ rank2_paper.pdf
   │  ├─ rank2_slides.pdf
   │  ├─ rank3_paper.pdf
   │  ├─ rank3_slides.pdf
   │  ├─ rank4_paper.pdf
   │  ├─ rank4_slides.pdf
   │  ├─ rank5_paper.pdf
   │  ├─ rank5_slides.pdf
   │  ├─ rank6_paper.pdf
   │  ├─ rank6_slides.pdf
   │  ├─ rank7_paper.pdf
   │  ├─ rank7_slides.pdf
   │  ├─ rank8_paper.pdf
   │  ├─ rank9_paper.pdf
   │  ├─ rank9_slides.pdf
   │  └─ slides_euro-neurips-overview.pdf
   └─ Paper
      ├─ .DS_Store
      ├─ Inventory Routing
      │  ├─ Bertsimas - 2019 - Dimitris Bertsimas.pdf
      │  ├─ Cui et al. - 2021 - Inventory Routing Problem under Uncertainty.pdf
      │  ├─ Deb et al. - 2002 - A fast and elitist multiobjective genetic algorith.pdf
      │  ├─ Kheiri - 2020 - Heuristic Sequence Selection for Inventory Routing.pdf
      │  ├─ Liu and Chen - 2011 - A heuristic method for the inventory routing and p.pdf
      │  ├─ Niakan and Rahimi - 2015 - A multi-objective healthcare inventory routing pro.pdf
      │  ├─ Partition
      │  │  ├─ Archetti et al. - 2007 - A Branch-and-Cut Algorithm for a Vendor-Managed In.pdf
      │  │  ├─ Bramel and Simchi-Levi - 2023 - A Location based Heuristic for General Routing Pro.pdf
      │  │  ├─ Campbell and Savelsbergh - 2004 - A Decomposition Approach for the Inventory-Routing.pdf
      │  │  ├─ Diabat et al. - 2021 - The Fixed-Partition Policy Inventory Routing Probl.pdf
      │  │  └─ Ekici et al. - 2015 - Cyclic Delivery Schedules for an Inventory Routing.pdf
      │  ├─ Peng and Murray - 2022 - VeRoViz A Vehicle Routing Visualization Toolkit.pdf
      │  ├─ Rahimi et al. - 2017 - Multi-objective inventory routing problem A stoch.pdf
      │  ├─ Singh et al. - 2015 - An incremental approach using local-search heurist.pdf
      │  ├─ Yu et al. - 2012 - Large scale stochastic inventory routing problems .pdf
      │  ├─ Zaitseva et al. - 2018 - Profit Maximization in Inventory Routing Problems.pdf
      │  ├─ Zhang et al. - 2022 - Learning-Based Branch-and-Price Algorithms for the.pdf
      │  ├─ genetic algorithm
      │  │  ├─ Aydın - 2014 - A Genetic Algorithm on Inventory Routing Problem.pdf
      │  │  ├─ Aziz and Mom - 2007 - Genetic algorithm based approach for the mufti pro.pdf
      │  │  ├─ Mahjoob et al. - 2022 - A modified adaptive genetic algorithm for multi-pr.pdf
      │  │  ├─ Moin et al. - 2011 - An efficient hybrid genetic algorithm for the mult.pdf
      │  │  ├─ Park et al. - 2016 - A genetic algorithm for the vendor-managed invento.pdf
      │  │  └─ Shukla et al. - 2013 - Genetic-algorithms-based algorithm portfolio for i.pdf
      │  └─ lagrangian relaxation
      │     ├─ Baldacci et al. - 2011 - New Route Relaxation and Pricing Strategies for th.pdf
      │     ├─ Balseiro et al. - 2021 - The Best of Many Worlds Dual Mirror Descent for O.pdf
      │     ├─ Golden et al. - 2008 - The Vehicle Routing Problem Latest Advances and N.pdf
      │     ├─ Hemmelmayr et al. - 2009 - Delivery strategies for blood products supplies.pdf
      │     ├─ Laporte - 1992 - The vehicle routing problem An overview of exact .pdf
      │     ├─ Pacheco et al. - 2012 - Optimizing vehicle routes in a bakery company allo.pdf
      │     ├─ Semet and Taillard - 1993 - Solving real-life vehicle routing problems efficie.pdf
      │     ├─ Shen et al. - 2011 - A Lagrangian relaxation approach for a multi-mode .pdf
      │     ├─ Vidal et al. - 2012 - A Hybrid Genetic Algorithm for Multidepot and Peri.pdf
      │     ├─ Yu et al. - 2006 - Large scale inventory routing problem with split d.pdf
      │     └─ Zhong and Aghezzaf - 2012 - MODELING AND SOLVING THE MULTI-PERIOD INVENTORY RO.pdf
      └─ Learning Method
         ├─ Achamrah et al. - 2022 - Solving inventory routing with transshipment and s.pdf
         ├─ Ahmedt-Aristizabal et al. - 2021 - Graph-Based Deep Learning for Medical Diagnosis an.pdf
         ├─ Bello et al. - 2017 - Neural Combinatorial Optimization with Reinforceme.pdf
         ├─ Bogyrbayeva et al. - 2022 - Learning to Solve Vehicle Routing Problems A Surv.pdf
         ├─ Chen and Tian - 2019 - Learning to Perform Local Rewriting for Combinator.pdf
         ├─ Dai et al. - 2018 - Learning Combinatorial Optimization Algorithms ove.pdf
         ├─ Gasse et al. - 2019 - Exact Combinatorial Optimization with Graph Convol.pdf
         ├─ Gong and Cheng - 2019 - Exploiting Edge Features in Graph Neural Networks.pdf
         ├─ Kool et al. - 2019 - Attention, Learn to Solve Routing Problems!.pdf
         ├─ Kwon et al. - 2021 - Matrix Encoding Networks for Neural Combinatorial .pdf
         ├─ Kwon et al. - Cost Shaping via Reinforcement Learning for Vehicl.pdf
         ├─ Li et al. - 2022 - Deep Reinforcement Learning for Solving the Hetero.pdf
         ├─ Nazari et al. - 2018 - Reinforcement Learning for Solving the Vehicle Rou.pdf
         ├─ Thakur and Peethambaran - 2020 - Dynamic Edge Weights in Graph Neural Networks for .pdf
         ├─ Veličković et al. - 2018 - Graph Attention Networks.pdf
         ├─ Vinyals et al. - Pointer Networks.pdf
         ├─ Xin et al. - 2021 - Step-Wise Deep Learning Models for Solving Routing.pdf
         ├─ Yang and Li - NENN Incorporate Node and Edge Features in Graph .pdf
         └─ Zhao et al. - 2021 - A Hybrid of Deep Reinforcement Learning and Local .pdf

```