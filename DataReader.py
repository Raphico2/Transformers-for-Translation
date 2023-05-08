from preprocess import load_data
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from HyperParameters import model_repository, path_data_repository, hyper_parameters
import torch
import spacy
import datasets
import re

MODEL_NAME = model_repository['MODEL_T5']
MODEL_SPACY = model_repository['MODEL_SPACY']


class DataReader:
    '''
    Class that load the data and perform the tokenization of the sentences, it divides each data set per sentences and also
    compute the attention mask for sentences (src and tgt) and root and modifers
    '''

    def __init__(self, max_seq_len, model_source_name=MODEL_NAME):

        self.max_seq_length = max_seq_len
        self.Tokenizer = AutoTokenizer.from_pretrained(model_source_name)
        self.nlp = spacy.load(MODEL_SPACY)
        self.train_path = None
        self.test_path = None
        self.val_path = None
        self.comp_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prefix = 'translate German to English: '
        self.prefix_for_root = ' Roots in English: '
        self.prefix_for_modifiers = '. Modifiers in English: '

    def load_train(self, path_to_file: str):
        self.train_path = path_to_file

    def load_test(self, path_to_file: str):
        self.test_path = path_to_file

    def load_val(self, path_to_file: str):
        self.val_path = path_to_file

    def load_comp(self, path_to_file: str):
        self.comp_path = path_to_file

    def compute_loaders(self, mode='normal'):

        if self.train_path != None:
            print("Currently Loading the train set...")
            train_dataset = load_data(self.train_path, tagged=True)
            self.train_dataset = self.create_dataset(train_dataset, tagged=True, mode=mode)
            print("Train set successfully loaded.")

        if self.test_path != None:
            print("Currently Loading the test set...")
            test_dataset = load_data(self.test_path, tagged=True)
            self.test_dataset = self.create_dataset(test_dataset,tagged=True, mode=mode)
            print("Test set successfully loaded")

        if self.val_path != None:
            print("Currently Loading the validation set...")
            val_dataset = load_data(self.val_path, tagged=False)
            self.val_dataset = self.create_dataset(val_dataset,tagged=False, mode=mode)
            print("Validation set successfully loaded")

        if self.comp_path != None:
            print("Currently Loading the competition set...")
            comp_dataset = load_data(self.comp_path, tagged=False)
            self.comp_dataset = self.create_dataset(comp_dataset,tagged=False, mode=mode)
            print("Competition set successfully loaded")

    def create_dataset(self, DS, tagged=True, mode='root_modifiers'):

        if tagged==True:
            dataset = {'input_ids': [], 'attention_mask': [], 'labels': []}

        else:
            dataset = {'input_ids': [], 'attention_mask': [], 'group': [], 'text_source':[]}

        for i in range(len(DS)):

            if tagged ==  True:
                text_source = DS[i]['text_german']
                text_target = DS[i]['text_english']
                roots, modifiers = self.extract_root_and_modifiers(text_target)

                if mode == 'root_modifiers':
                    inputs = self.prefix + text_source + self.prefix_for_root + roots + self.prefix_for_modifiers + modifiers
                else:
                    inputs = self.prefix + text_source

                print(inputs)
                text_source_dict = self.Tokenizer(inputs, return_tensors="pt", max_length=self.max_seq_length,
                                                  truncation=True, padding="max_length", add_special_tokens=True)

                with self.Tokenizer.as_target_tokenizer():
                    text_target_dict = self.Tokenizer(text_target, return_tensors="pt", max_length=self.max_seq_length,
                                                      truncation=True, padding="max_length", add_special_tokens=True)

                # get the tokens ids
                text_source_ids = text_source_dict["input_ids"].squeeze(0)
                text_target_ids = text_target_dict["input_ids"].squeeze(0)
                # compute les mask d'attentions
                attention_mask_src = text_source_dict["attention_mask"].squeeze(0)

                # register in a dataset
                dataset['input_ids'].append(text_source_ids)
                dataset['attention_mask'].append(attention_mask_src)
                dataset['labels'].append(text_target_ids)


            else:
                text_source = DS[i]['text_german']
                roots = DS[i]['root']
                modifiers = DS[i]['modifiers']

                if mode == "root_modifiers":
                    inputs = self.prefix + text_source + self.prefix_for_root + roots + self.prefix_for_modifiers + modifiers
                else:
                    inputs = self.prefix + text_source

                group = DS[i]['group']

                text_source_dict = self.Tokenizer(inputs, return_tensors="pt", padding="max_length",
                                                  max_length=self.max_seq_length, add_special_tokens=True)

                text_source_ids = text_source_dict["input_ids"].squeeze(0)
                attention_mask_src = text_source_dict["attention_mask"].squeeze(0)

                dataset['input_ids'].append(text_source_ids)
                dataset['attention_mask'].append(attention_mask_src)
                dataset['group'].append(group)
                dataset['text_source'].append(text_source)

        dataset = datasets.Dataset.from_dict(dataset)
        dataset.set_format('torch')
        return dataset

    def extract_root_and_modifiers(self, text):
        '''
        Extract the root and the modifiers from the english sentences in order to fit the training and test set
        :param sentence: Sentence in english
        :return: The root and modifiers of the sentence
        '''

        mod_string =''
        roots = ''
        for sentence in self.nlp(text).sents:
            # Extraire la racine de la phrase
            root = None
            for word in sentence:
                if word.dep_ == "ROOT":
                    root = word
                    break
            if root is None:
                continue

            modifiers = []
            for child in root.children:
                if child.dep_ != "punct":
                    modifiers.append(child)
                if len(modifiers) == 2:
                    break
            if len(modifiers) == 2:
               mod_string += '(' + str(modifiers[0]) + ',' + str(modifiers[1]) + '), '
            elif len(modifiers) == 1:
               mod_string += '(' + str(modifiers[0]) + '), '
            else:
                mod_string += '(-), '

            if root is None:
                roots += '-, '

            else:
                roots += str(root) + ', '
        return roots[:-2], mod_string[:-2]


if __name__ == '__main__':

    #check is the Reader class is working properly
    Reader = DataReader(hyper_parameters['max_seq_len'])
    Reader.load_train(path_data_repository['PATH_TRAIN'])
    Reader.compute_loaders(mode='root_modifiers')
    print(Reader.train_dataset[0]['input_ids'])