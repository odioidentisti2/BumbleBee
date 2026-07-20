muta = {
    'task': 'binary_classification',
    'path': 'DATASETS/MUTA_SARPY_4204.csv',
    'smiles_header': 'SMILES',
    'target_header': 'Experimental_value',
    'tox_map': {'Mutagenic': 1, 'NON-Mutagenic': 0},
    # Split info (optional)
    'split_header': 'Status',
    'train_split': 'Training',
    'test_split': 'Test',
    # 'split_map': {'train': 'Training', 'test': 'Test'}
}

muta_train = {
    'task': 'binary_classification',
    'path': 'DATASETS/MUTA_train.csv',
    'id_header': 'Id',
    'smiles_header': 'SMILES',
    'target_header': 'Experimental_value',
    'tox_map': {'Mutagenic': 1, 'NON-Mutagenic': 0},
}

muta_test = {
    'task': 'binary_classification',
    'path': 'DATASETS/MUTA_test.csv',
    'id_header': 'Id',
    'smiles_header': 'SMILES',
    'target_header': 'Experimental_value',
    'tox_map': {'Mutagenic': 1, 'NON-Mutagenic': 0},
}

logp = {
    'task': 'regression',
    'path': 'DATASETS/LogP.csv',
    'smiles_header': 'SMILES',
    'target_header': 'Experimental value',
}

logp_split = {
    'task': 'regression',
    'path': 'DATASETS/logp_split.csv',
    'smiles_header': 'SMILES',
    'target_header': 'Experimental value',
    # Split info (optional)
    'split_header': 'Status',
    'train_split': 'Training',
    'test_split': 'Test',
    # 'split_map': {'train': 'Training', 'test': 'Test'}
}