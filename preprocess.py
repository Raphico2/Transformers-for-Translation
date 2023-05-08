from HyperParameters import hyper_parameters

def load_data(path_to_file : str, tagged=True):
    """Load the sentences from the files
    path_to_file (str): the path to the file
    tagged (Boolean): indiate if the file has the translation in english or just the german sentences
    Returns:
        list: list of dictionary containing the sentences in english and their translation in german
    """
    Dataset = []
    counter = 0
    German = True

    if tagged == True:
        sample = {"text_german": '', 'text_english': '', 'group': counter}
    else:
        sample = {"text_german": '', 'root': '', 'modifiers': '', 'group': counter}

    with open(path_to_file,  encoding="utf8") as f:
        for line in f:
            splited_line = line.split()
            if len(splited_line) == 0:
                Dataset.append(sample)
                counter+=1
                if tagged == True:
                    sample = {"text_german": '', 'text_english': '', 'group': counter}
                else:
                    sample = {"text_german": '', 'root': '', 'modifiers': '', 'group': counter}

            elif splited_line[0] =="German:":
                German = True
            elif splited_line[0] =="English:":
                German = False

            elif splited_line[0] == 'Roots' and splited_line[1] == 'in' and splited_line[2] == 'English:' and tagged==False:
                 roots = ' '.join(splited_line[3:])
                 sample['root'] += roots

            elif splited_line[0] == 'Modifiers' and splited_line[1] == 'in' and splited_line[2] == 'English:' and tagged==False:
                 modifiers = ' '.join(splited_line[3:])
                 sample['modifiers'] += modifiers

            else:
                if German == True:
                    sample['text_german'] += line[:-1] + ' '

                else:
                    sample['text_english'] += line[:-1] + ' '

    return Dataset


if __name__ == '__main__':
    #check if the function load data work properly
    ds = load_data(hyper_parameters['PATH_TRAIN'], tagged=True)
    print(ds[0])




