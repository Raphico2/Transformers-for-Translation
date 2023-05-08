import torch
import numpy as np
import evaluate
from DataReader import DataReader
from utils import recompute_sentence
from HyperParameters import hyper_parameters
import sacrebleu

TARGET_VOCAB_SIZE = hyper_parameters['vocab_size']
EMBED_DIM = hyper_parameters['embedding_dim']


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(tagged_en, true_en):
    '''
    Compute the blue score using sacrebleu
    :param tagged_en:
    :param true_en:
    :return:
    '''
    metric = evaluate.load("sacrebleu")
    # metric = evaluate.load("accuracy")
    tagged_en = [x.strip().lower() for x in tagged_en]
    true_en = [x.strip().lower() for x in true_en]
    result = metric.compute(predictions=tagged_en, references=true_en)
    result = result['score']
    result = round(result, 2)
    return result


def compute_BLEU_score(pred, label):
    '''
    Other way computing the BLEU score
    :param pred:
    :param label:
    :return:
    '''
    bleu_score = sacrebleu.corpus_bleu(pred, [label]).score
    return bleu_score


def read_file(file_path):
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str.strip())
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line + ' '
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def calculate_score(file_path1, file_path2):
    file1_en, file1_de = read_file(file_path1)
    file2_en, file2_de = read_file(file_path2)
    for sen1, sen2 in zip(file1_de, file2_de):
        if sen1.strip().lower() != sen2.strip().lower():
            raise ValueError('Different Sentences')
    score = compute_metrics(file1_en, file2_en)
    return score


def Evaluate_on_test(Reader: DataReader, model, device, print_sentence=True):
    '''
    Compute the Blue score on the test set
    :param Reader: The reader of the project
    :param model: The model we use to predict
    :param device:
    :return: The mean Blue score on the test Set
    '''
    Blue_score = []
    print('test the model on the test dataset')
    for i in range(len(Reader.test_dataset)):

        input_token_ids = Reader.test_dataset[i]['input_ids'].to(device)
        input_token_ids = input_token_ids.unsqueeze(0)
        attention_mask = Reader.test_dataset[i]['attention_mask'].to(device)
        attention_mask = attention_mask.unsqueeze(0)
        labels = Reader.test_dataset[i]['labels'].to(device)
        pred_ids = model.generate(input_ids=input_token_ids,
                                  attention_mask=attention_mask,
                                  decoder_start_token_id=model.config.decoder_start_token_id,
                                  num_beams=hyper_parameters["beam_size"],
                                  max_length=hyper_parameters["max_length"],
                                  early_stopping=True)
        predicted_text = recompute_sentence(pred_ids.squeeze(0), Reader.Tokenizer)
        true_text = recompute_sentence(labels, Reader.Tokenizer)

        if print_sentence == True and i % 200 == 0:
            print(predicted_text)
            print(true_text)

        BLUE = compute_BLEU_score(predicted_text, true_text)
        Blue_score.append(BLUE)
    return np.mean(Blue_score)
