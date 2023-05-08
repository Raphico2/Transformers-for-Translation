import torch
import matplotlib.pyplot as plt

def recompute_sentence(token_ids, tokenizer):
    '''
    Reconstitue the sentence from the tokens
    :param predictions:
    :param tokenizer:
    :return: the reconstituted first senten
    '''
    output_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return output_text


def plot_graph(x, y, title, xlabel, ylabel, color='red'):
    '''
    Plot graph, available for accuracy score and loss score in function of epochs
    :param x:
    :param y:
    :param title:
    :param xlabel:
    :param ylabel:
    :param color:
    '''
    plt.plot(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def save_model(model, path_to_model):
    '''
    Save the weights of the Transformers model
    :param model:
    :param path_to_model:
    :return:
    '''
    print('Saving the model...')
    torch.save(model.state_dict(), path_to_model)
    print('Model correctly saved')



def load_model(model, path_to_model, device):
    '''
    Load the weight of the model
    :param model:
    :param path_to_model:
    :return: the model with the right weights
    '''
    print('Loading the model...')
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model = model.to(device)
    print('Model correctly loaded')
    return model


def write_result_into_file(src_dictionary, trg_dictionary, file_path):
    '''
    Write the result of the model into the comp files
    :param src_dictionary:
    :param trg_dictionary:
    :param file_path:
    :return: None
    '''

    german_line = 'German:'
    english_line = 'English:'

    idx = 0
    length = len(src_dictionary)
    with open(file_path, 'w', encoding="utf-8") as f:
        while (idx < length):
            f.write(german_line + "\n")
            f.write(src_dictionary[idx]+"\n")
            f.write(english_line + "\n")
            f.write(trg_dictionary[idx] +"\n")
            f.write("\n")
            idx += 1

