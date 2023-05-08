import utils
import torch
import numpy as np
from Evaluate import recompute_sentence, Evaluate_on_test, postprocess_text
from sacrebleu import corpus_bleu
from DataReader import DataReader
from HyperParameters import hyper_parameters, path_data_repository, model_repository
from transformers import get_linear_schedule_with_warmup, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, IntervalStrategy
from datasets import load_dataset, load_metric
from utils import save_model


tokenizer = AutoTokenizer.from_pretrained(model_repository["MODEL_T5"])
metric = load_metric("sacrebleu")

def compute_metrics_train(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds, labels = postprocess_text(preds, labels)

    result = metric.compute(predictions=preds, references=labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["Max_sen_length"] = np.max(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



def train(Data_reader: DataReader, model, optimizer, scheduler):

    parameters = Seq2SeqTrainingArguments(
                    output_dir = '/content/drive/MyDrive/Models/myModel',
                    evaluation_strategy="epoch",
                    save_strategy=IntervalStrategy.EPOCH,
                    learning_rate=hyper_parameters['learning_rate'],
                    per_device_train_batch_size=hyper_parameters['batch_size_train'],
                    per_device_eval_batch_size=hyper_parameters['batch_size_test'],
                    weight_decay=0.01,
                    save_total_limit=1,
                    num_train_epochs=hyper_parameters['epochs'],
                    prediction_loss_only=False,
                    predict_with_generate=True,
                    fp16=True,
                    load_best_model_at_end=True,
                    generation_max_length=hyper_parameters['max_length'],
                    greater_is_better=True,
                    metric_for_best_model='eval_bleu',
                )

    data_collator = DataCollatorForSeq2Seq(Data_reader.Tokenizer, model=model)

    trainer = Seq2SeqTrainer(
                    model=model,
                    args=parameters,
                    train_dataset=Data_reader.train_dataset,
                    eval_dataset=Data_reader.test_dataset,
                    data_collator=data_collator,
                    tokenizer=Data_reader.Tokenizer,
                    compute_metrics=compute_metrics_train,
                    optimizers=[optimizer, scheduler]
                )

    trainer.train()




if __name__ == '__main__':

    Reader = DataReader(hyper_parameters['max_seq_len'])
    Reader.load_train(path_data_repository['PATH_TRAIN'])
    Reader.load_test(path_data_repository['PATH_TEST'])
    Reader.compute_loaders(mode=hyper_parameters["mode"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_repository['MODEL_T5'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_parameters['learning_rate'], eps=hyper_parameters['epsilon'])
    num_training_steps = len(Reader.train_dataset) // hyper_parameters["batch_size_train"]*hyper_parameters["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    model = model.to(device)

    train(Reader, model, optimizer, scheduler)
    save_model(model, path_data_repository['PATH_MODEL'])

