# Transformers-for-Translation
# Using Transformers model for Translation of german text to english

We create a model for translation between sentences from German to English. The Train and Validation set that were labeled had only sentences in German and their translation in English. On the other side, the Validation and Competition files unlabeled contained German sentences, the root and 2 modifiers of the root from the translated sentence in English.
For the task we decided to implement a pre-trained transformers model that we modified to get a translation very close to the target text.

# Preprocess :

The preprocess is made with the help of two Python file:

- Preprocess.py : contains all the auxiliary functions to compute embeddings et load the data

- DataReader.py : contains the Data Reader class that will iterate over the batch of data and structures the data pipeline.

Before collecting the sentences, we made a preprocess on our training and validation set to keep only sentences well translated. We deleted the German sentences that were translated in Chinese instead of English. Furthermore, we decided to work with text and not sentences. Because we will use a transformer, it can deal with a very large text and not only sentence. Moreover, the number of sentences in English and in German is not the same that makes complicated to work with sentences.
For the labeled files, the data is loaded and divided into text containing many sentences. For each English text (target text) we extracted the root and two of the root’s modifiers for each sentences with the help of the library spacy.

There is two different mode, the mode that includes the root and the modifiers, and the mode which do not take in consideration the root and modifiers. The modification of the mode can be done directly in the hyperParameters.py.

Then we construct our input text by writing :
- “Translate German to English” {Text in German}
- “Root in English {roots}” (only if mode=’root_modifier’)
- “Modifiers in English {(modifier1, modifier2)...} as modifiers. (Only if
mode=’root_modifier’)
Our target text remains the English translated text itself.
     
    
For each text we applied a tokenization based on the pre-trained model t5-base. The tokenization is made on sub-word and is multilingual, that can help a lot for translation problems. We applied a target tokenization for the target texts.
We applied the same preprocess for the 4 different files.
At the end of the preprocess, the object Reader contains sample that include 3 features. The first one is called the source text, containing the tokenized text in German with the prefix (and the root and modifiers If the mode is activated), the second one and the padding mask of the source text and the last one contains the target texts (in token ids).


# Models


Our model is a transformer, pre-trained with T5 base with the module T5ForConditionalGeneration. The transformer receives as input the token ids of our input sentence. The first part of the input is the instruction: “Translate German to English” following by the German text. The second part of the input is the description of the root and modifiers. The output will be the target sequence.
The tokenized input sentence enters the transformer and is converted into embedding (T5 embedding). Then the input sentence enters in the Encoder, the output of the encoder is given to the decoder at the same time of the labels and the causal mask.
The transformer contains an embedding step based of T5 embedding with an embedding dimension of 768 which is also cumulated with a positional embedding.
We trained our data with a batch size of 16, a learning rate of 5e-5 and 8 epochs. All the hyperparameters are available in the file hyperparameters.py.
The speed of the training phase was approximatively 10 minutes per epochs using the SeqToSeqTrainer from the module transformers of Hugging Face. It takes 2 minutes to compute the evaluation after each epoch ends with a GPU.
After each epoch, we computed the BLEU score on the test set (Val.labeled) and we kept the model that gave us the best BLEU score.
During the training step, we experienced two different methods, the first one with the root and modifiers, the second one without the root and modifiers, we used the same model (T5ForConditionalGeneration with T5-base) but our input ids changed by modifying the mode in the preprocess.


# Results and predictions


Results of the training and evaluation step:

The model that gives the best BLUE score on the validation set is the model that includes the root and the modifiers, with a BLEU score of 57.18 on the validation set and 88.32 on the train set. In order to generate the translation of the unlabeled files we will use this model and take in consideration the root and modifiers for each sentence.
Our model has a forward function but also a translate function. The first one was used for training, the second one allows the model to translate a sentence without providing the target sentence. We used the Beam search algorithm with a beam size of 4 to get the best translation in term of probability. Exactly as for the training part, the sentence in German (in token ids) is sent to the model with the fine tunned sentences and the root and modifiers, it generates a translation using the beam search and return the best translation.
We can see the Evaluate function inside the file Evaluate.py. Moreover, we implemented a file named Test.py that run the function generate() of our model on the Val.labeled file and return the BLUE score.

# Limits and improvement:

We discussed about several improvements that could be done in order to improve our model. The first one would be to train our model with sentences instead of text. The attention head would better focus on the root and modifiers with only one sentence to translate each time. We could, if we had more powerful GPU with better memory use greater pre-trained model such as T5-11B which has been trained on many more data.
We also would divide our project into two part, a translation part and a reformulation part. We would use 2 transformers and train our model to first translate the sentence from German to English and then reformulate the sentence with the modifiers and the root.
There are several other model that could be tried but it needs more powerful GPU.


Link to the model :
https://drive.google.com/drive/folders/15IfFZ668rpMs1V9HMXyC5S_5jIACHE7Q?usp=share _link
