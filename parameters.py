GLOB = {

    ## Training
    'lr': 1e-4,
    'epochs': 100,
    'batch_size': 32,
    'random_seed': 42,
    # 'weight_decay': 1e-5,

    ## Model
    'layer_types': ['M', 'M', 'S', 'P'],
    'in_out_mlp': 128,

    ## ESA
    'hidden_dim': 128,
    'mlp_expansion': 2,
    'heads': 8,
    'seeds': 32,

    # Regularization
    'ESA_dropout': 0.0,
    'SAB_dropout': 0.0,
    'PMA_dropout': 0.0,
}