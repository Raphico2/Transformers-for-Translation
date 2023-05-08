import torch
from utils import recompute_sentence, write_result_into_file
from Transformers import GermanEnglishTranslatorWithRootModifers
from DataReader import DataReader
from HyperParameters import hyper_parameters, path_data_repository, model_repository
from Evaluate import Evaluate_on_test
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import load_model


if __name__ == '__main__':
    Reader = DataReader(hyper_parameters['max_length'])
    Reader.load_test(path_data_repository['PATH_TEST'])
    Reader.compute_loaders()


    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    model = AutoModelForSeq2SeqLM.from_pretrained(path_data_repository['PATH_MODEL_TRAINED'])
    model = model.to(device)
    Bleu_score = Evaluate_on_test(Reader, model, device, print_sentence=True)
    print("the Bleu score for the model on the Val.labeled dataset is : " + str(Bleu_score))