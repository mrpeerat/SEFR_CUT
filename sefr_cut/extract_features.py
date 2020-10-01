# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, \
    Concatenate, Flatten, SpatialDropout1D, \
    BatchNormalization, Conv1D, Maximum, ZeroPadding1D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
import ahocorasick
from urllib.request import urlopen
import os
PATH = os.path.dirname(__file__)

with open(os.path.join(PATH, 'variable','words_modified.txt'),'r',encoding='utf-8-sig') as f:
    dict_ = f.read().strip().split('\n')

A = ahocorasick.Automaton()
for idx, word in enumerate(dict_):
  # insert word, with length of word
  # this will be used later to calculate start_index and end_index (as features)
    A.add_word(word, len(word)) 

A.make_automaton()

# Character types adapted from Haruechaiyasak et al. 2008.

# Character that can be the final consonant in a word
chartype_c = '\u0e01\u0e02\u0e03\u0e04\u0e06\u0e07\u0e08\u0e0a\u0e0b\u0e0d\u0e0e\u0e0f' + \
  '\u0e10\u0e11\u0e12\u0e13\u0e14\u0e15\u0e16\u0e17\u0e18\u0e19\u0e1a\u0e1b\u0e1e\u0e1f' + \
  '\u0e20\u0e21\u0e22\u0e23\u0e24\u0e25\u0e26\u0e27\u0e28\u0e29\u0e2a\u0e2c\u0e2d'

# Character that cannot be the final consonant in a word
chartype_n = '\u0e05\u0e09\u0e0c\u0e1c\u0e1d\u0e2b\u0e2e'

# Vowel that cannot begin a word
chartype_v = '\u0e30\u0e31\u0e32\u0e33\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e45\u0e47'

# Vowel that can begin a word
chartype_w = '\u0e40\u0e41\u0e42\u0e43\u0e44'

# Combining symbol
chartype_s = '\u0e3a\u0e4c\u0e4d\u0e4e'

# Standalone symbol
chartype_a = '\u0e2f\u0e46\u0e4f\u0e5a\u0e5b'

# Tone marks
chartype_t = '\u0e48\u0e49\u0e4a\u0e4b'

# Digit character
chartype_d = '0123456789\u0e50\u0e51\u0e52\u0e53\u0e54\u0e55\u0e56\u0e57\u0e58\u0e59'

# Currency character
chartype_b = '$฿'

# Quote character
chartype_q = '\'\"'

# Other character
chartype_o = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Space character inside a word
# chartype_p

# Space character
chartype_z = ' \u00a0'

# Undefined
# chartype_x

tags = [
    ('c', chartype_c),
    ('n', chartype_n),
    ('v', chartype_v),
    ('w', chartype_w),
    ('s', chartype_s),
    ('a', chartype_a),
    ('t', chartype_t),
    ('d', chartype_d),
    ('b', chartype_b),
    ('q', chartype_q),
    ('o', chartype_o),
    ('z', chartype_z)
]

# Character type
def get_ctype(c):
    for tag in tags:
        if c in tag[1]:
            return tag[0]
    return 'x'

def extract_features_crf(doc,y_entropy_var,y_prob_var):
    doc_features = []
    
    # Get (start, end) candidates from dictionary
    dict_start_boundaries = set()
    dict_end_boundaries = set()
    for end_index, length in A.iter(doc):
        start_index = end_index - length + 1
        dict_start_boundaries.add(start_index)
        dict_end_boundaries.add(end_index)

    for i, char in enumerate(doc):
        char_features = {}
        char_features = {
            "bias":'b',
            'char': char,
            'entropy' : y_entropy_var[i],
            'prob' : y_prob_var[i][1],
        }
  
        if i == 0:
            char_features.update({
                "start":True
            })
        else:
            char_features.update({
                "start":False
            })
        if i == len(doc)-1:
            char_features.update({
                "end":True
            })
        else:
            char_features.update({
                "end":False
            })

        back_ward = 4
        for_ward = 2
        if i < back_ward:
            for index in range(back_ward-i,0,-1):
                char_features.update({
                    f"char_[-{index+i}]" : ' ',
                    f'ctype[-{index+i}]' : get_ctype(' '),
                }) 
        for index in range(0,back_ward,1):
            try:
                char_features.update({
                    f"char_[-{index+1}]" : doc[i-index-1],
                    f'ctype[-{index+1}]' : get_ctype(doc[i-index-1]),
                })
            except:
                continue
        
        text=doc[i+1:i+for_ward+1] # forward
        while(len(text)<for_ward):
            text+=' '
        for index,char_ in enumerate(text):
            char_features.update({
                f"char_[+{index+1}]" : char_,
                f'ctype[+{index+1}]' : get_ctype(char_),
            })

        # If this character can be a start of word, according to our dictionary
        
        if i in dict_start_boundaries:
            char_features.update({
                "dict_start":True
            })
        else:
            char_features.update({
                "dict_start":False
            })
        
        # If this character can be an end of word, according to our dictionary
        
        if i in dict_end_boundaries:
            char_features.update({
                "dict_end":True
            })
        else:
            char_features.update({
                "dict_end":False
            })
#         len_features = (for_ward*2)+(back_ward*2)+2+5
#         if len(char_features) < len_features:
#             print(len(char_features),char_features)
        
        doc_features.append([char_features])       
    return doc_features  

def conv_unit(inp, n_gram, no_word=200, window=2):
    out = Conv1D(no_word, window, strides=1, padding="valid", activation='relu')(inp)
    out = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(out)
    out = ZeroPadding1D(padding=(0, window - 1))(out)
    return out

def get_convo_nn2(no_word=200, n_gram=21, no_char=178):
    input1 = Input(shape=(n_gram,))
    input2 = Input(shape=(n_gram,))

    a = Embedding(no_char, 32, input_length=n_gram)(input1)
    a = SpatialDropout1D(0.15)(a)
    a = BatchNormalization()(a)

    a_concat = []
    for i in range(1,9):
        a_concat.append(conv_unit(a, n_gram, no_word, window=i))
    for i in range(9,12):
        a_concat.append(conv_unit(a, n_gram, no_word - 50, window=i))
    a_concat.append(conv_unit(a, n_gram, no_word - 100, window=12))
    a_sum = Maximum()(a_concat)

    b = Embedding(12, 12, input_length=n_gram)(input2)
    b = SpatialDropout1D(0.15)(b)

    x = Concatenate(axis=-1)([a, a_sum, b])
    #x = Concatenate(axis=-1)([a_sum, b])
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2], outputs=out)
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy', metrics=['acc'])
    return model

