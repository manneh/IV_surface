TRAINING_PARAMS = {
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "mse_coef": 0.05,
    "mlse_coef": 0.4,
    "mspe_coef": 0.55,
    "epochs": 30000, 
    "writing_dir": 'ninjatiny', 
    'display_step': 200,
    'model_saving_path': 'colfinal', 
    'writing_step' : 20
}

MODEL_PARAMS = {
    "hidden_sizes_list": [4, 16, 120, 512, 64],
}
