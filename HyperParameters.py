"""
The Hyper Parameter file give access to all of the parameters of the models, if we need to execute one change it must be done here
"""

hyper_parameters = {'num_head': 12,
                    'learning_rate': 2e-5,
                    'batch_size_train': 16,
                    'batch_size_test': 16,
                    'epochs': 20,
                    'dropout': 0.1,
                    'epsilon': 1e-6,
                    'hidden_dim': 2048,
                    'num_layer': 4,
                    'max_seq_len': 400,
                    'max_r_m_len': 10,
                    'embedding_dim': 768,
                    'beam_size': 3,
                    'vocab_size': 32000,
                    'mode': 'root_modifiers'}

path_data_repository = {'PATH_TRAIN': 'train.labeled',
                        'PATH_TEST': 'val.labeled',
                        'PATH_VAL': 'val.unlabeled',
                        'PATH_COMP': 'comp.unlabeled',
                        'PATH_MODEL': "Models/transformer.pt",
                        'PATH_RESULT_VAL': 'results/val_345237721_342602281.labeled',
                        'PATH_RESULT_COMP': 'results/comp_345237721_342602281.labeled',
                        'PATH_MODEL_TRAINED': 'https://drive.google.com/drive/folders/15IfFZ668rpMs1V9HMXyC5S_5jIACHE7Q?usp=share _link'}

model_repository = {'MODEL_T5': 't5-base',
                    'MODEL_SPACY': 'en_core_web_sm'}


