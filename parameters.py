from pprint import pprint


parameters = []

# PARAMETERS
main_params = {
    'batch_size': 32,
    'random_seed': 42,
}
parameters.append(main_params)

trainer_params = {
    'lr': 1e-4,
    'epochs': 100,
    # 'weight_decay': 1e-5,
    'inject': False,  # Enable baseline injection
}
parameters.append(trainer_params)

model_params = {
    'layer_types': ['M', 'M', 'M', 'P'],
    'in_out_mlp': 128,
    'hidden_dim': 128,
    'mlp_expansion': 2,
    'ESA_dropout': 0.0,
    'BATCH_DEBUG': False,  # Debug: use batch Attention even on CPU
}
parameters.append(model_params)

attention_params = {
    'heads': 8,
    'seeds': 1,
    'SAB_dropout': 0.0,
    'PMA_dropout': 0.0,
}
parameters.append(attention_params)


def print_parameters():
    print('\nPARAMETERS:')
    for dict in parameters:
        pprint(dict)



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