import os
import numpy as np


# LANGUAGES
english_tokens = "abcdefghijklmnopqrstuvwxyz \n\t"
french_tokens = "abcdefghijklmnopqrstuvwxyz àéèëô\n\t"

####################################################################### 
# EXTRACT DATA: file --> python text obj                              #
# data source: https://www.isi.edu/natural-language/download/hansard/ #
# this should not have to be called more than once                    #
#######################################################################

num_train = 10000
num_valid = 500
num_test = 500

def extract_data(path, num):
    """
    Will put (english, french) line pairs in the lst from path.
    """
    lst = []
    nl = 0 # num lines
    for s in os.listdir(path):
        if nl >= num: break
        s = s.split(".")
        # skip all non english files to not double count pairs
        if s[len(s)-1] != 'e': continue
        # get the index name without the suffix
        s = s[:len(s)-1]
        s = '.'.join(s)
        # read the files
        e_file = open(path+s+".e", encoding="ISO-8859-1")
        f_file = open(path+s+".f", encoding="ISO-8859-1")
        et, ft = e_file.read(), f_file.read()
        # add individual line pairs to the data set
        for el, fl in zip(et.split("\n"), ft.split("\n")):
            nl += 1
            lst.append(("\t"+el+"\n", "\t"+fl+"\n"))
    return lst

def filter_by_language(data):
    """
    Assumes data is the format outputted by extract_data (english, french pairs)
    Will remove any characters from data that's not in desired language. Converts upper to lower before filter.
    """
    res = []
    for el, fl in data:
        # convert to lower case
        el, fl = el.lower(), fl.lower()
        # filter through the languages
        el_filter = ''.join([c for c in el if c in list(english_tokens)])
        fl_filter = ''.join([c for c in fl if c in list(french_tokens)])
        res.append((el_filter, fl_filter))
    return res

##########################################################
# DATA UTILS: python text obj --> one hot encoded arrays #
##########################################################

def load_data(path = './data/senate/', small=True):
    """
    path: path to where the stored .npy files are.
    small: if True, will load the small versions of the files.
    returns: train, valid, test (np arrays)
    """
    if small: sm = '_small'
    else: sm = ''
    train_path = path + "train" + sm + ".npy"
    valid_path = path + "valid" + sm + ".npy"
    test_path = path + "test" + sm + ".npy"
    train, valid, test = np.load(train_path), np.load(valid_path), np.load(test_path)
    return train, valid, test

def one_hot(data, enc_alphabet=english_tokens, dec_alphabet=french_tokens, reverse=False):
    """
    Given raw text lines (i.e. from the output of load_data), will return a one hot encoded
    np array that can be given to a keras fit method.
    Assumes that the encoder will be trained on the first element of each tuple.
    """
    # computes reverse token indicies
    enc_token_index = {v:k for k,v in enumerate(enc_alphabet)}
    dec_token_index = {v:k for k,v in enumerate(dec_alphabet)}
    
    
    def string_vectorizer(strng, alphabet):
        vector = [[0 if char != letter else 1 for char in alphabet] 
                      for letter in strng]
        return np.array(vector)
    
    num_strings = data.shape[0]
    max_enc_len = max([len(l[0]) for l in data])
    max_dec_len = max([len(l[1]) for l in data])
    enc_arr = np.zeros((num_strings, max_enc_len, len(enc_alphabet)), dtype='float32')
    dec_arr = np.zeros((num_strings, max_dec_len, len(dec_alphabet)), dtype='float32')
    dec_target_arr = np.zeros((num_strings, max_dec_len, len(dec_alphabet)), dtype='float32')
    
    for i, (input_text, target_text) in enumerate(data):
        for t, char in enumerate(input_text):
            enc_arr[i, t, enc_token_index[char]] = 1.
        enc_arr[i, t + 1:, enc_token_index[' ']] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            dec_arr[i, t, dec_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                dec_target_arr[i, t - 1, dec_token_index[char]] = 1.
        dec_arr[i, t + 1:, dec_token_index[' ']] = 1.
        dec_target_arr[i, t:, dec_token_index[' ']] = 1.
    
    # does reversal
    if reverse:
        enc_arr = enc_arr[:, ::-1, :]
        
    return enc_arr, dec_arr, dec_target_arr

# exposes the datasets
train_raw, valid_raw, test_raw = load_data()
train, valid, test = one_hot(train_raw, reverse=True), one_hot(valid_raw, reverse=True), one_hot(test_raw, reverse=True)
