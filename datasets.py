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
    'split_map': {'train': 'Training', 'test': 'Test'}
}

logp = {
    'task': 'regression',
    'path': 'DATASETS/LogP.csv',
    'smiles_header': 'SMILES',
    'target_header': 'Experimental value',
}

logp_split = {
    'task': 'regression',
    'path': 'DATASETS/Logp_split.csv',
    'smiles_header': 'SMILES',
    'target_header': 'Experimental value',
    # Split info (optional)
    'split_header': 'Status',
    'train_split': 'Training',
    'test_split': 'Test',
}