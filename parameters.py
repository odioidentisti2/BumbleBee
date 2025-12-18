GLOB = {

    ## Training
    'lr': 1e-4,
    'epochs': 100,
    'batch_size': 32,
    'random_seed': 30,
    # 'weight_decay': 1e-5,

    ## Model
    'layer_types': ['M', 'M', 'M', 'P'],
    'in_out_mlp': 128,

    ## ESA
    'hidden_dim': 128,
    'mlp_expansion': 2,
    'heads': 8,
    'seeds': 1,

    # Regularization
    'ESA_dropout': 0.0,
    'SAB_dropout': 0.0,
    'PMA_dropout': 0.0,
}


    ## ESA: README
    # lr = 0.0001
    # BATCH_SIZE = 128
    # HIDDEN_DIM = 256  (= graph_dim)
    # MLP_hidden_dim = 256 (graph-level) or 512 (node-level)
    # NUM_HEADS = 16  (= 4 in the template)
    # LAYER_TYPES = 'MMSP' default; 'MSMSMP' graph-level
    # DROPOUT = 0 !!!!!!!!!
    # weight_decay = 1e-10 nel README (useless)
    ## ESA (hardcoded)
    # PMA seeds = 32
    # model's MLP hidden dimension = 128
    # model's MLP dropout = 0