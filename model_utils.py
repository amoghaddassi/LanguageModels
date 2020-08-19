from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional

###################################
# GENERIC MODEL UTILITY FUNCTIONS #
###################################

def fit_model(model, filename, train_enc, train_dec, train_dec_target,
             batch_size = 64, epochs = 10, validation_split = .2):
    """
    Assumes that model maps encoder to decoder text, as generated by data_utils
    filename is where the model will be saved
    """
    # fits the model
    model.fit([train_enc, train_dec], train_dec_target,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split)
    # saves the model
    model.save(filename)

def translate(encoder_model, decoder_model, input_seq, input_len, dec_alphabet):
    """
    target = encode(src)
    while target not done:
        c = decode(target)
        target += c
    return target
    """
    # language stuff
    dec_token_index = {v:k for k,v in enumerate(dec_alphabet)}
    num_dec_tokens = len(dec_alphabet)
    max_decoder_len = input_len
    # input --> hidden state
    states_value = encoder_model.predict(input_seq)
    
    # generate empty target sequence with offset
    target_seq = np.zeros((1, 1, num_dec_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, dec_token_index['\t']] = 1.
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = dec_alphabet[sampled_token_index]
        decoded_sentence += sampled_char
        # check stop condition
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_len:
            stop_condition = True
        # update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # update states
        states_value = [h, c]
    return decoded_sentence

######################################################    
# MODEL DEFINITION FUNCTIONS: return compiled models #
######################################################
    
def basic_lstm(num_enc_tokens, num_dec_tokens, latent_dim=128, 
                optimizer='rmsprop', loss='categorical_crossentropy'):
    """
    One of the simplest models that can be implemented for this task. Has following architecture:
    - encoder: inputs --> hidden_state using lstm
    - decoder: hidden_state --> target using lstm
    Based on: https://arxiv.org/pdf/1409.3215.pdf
    Tutorial: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    """
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_enc_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_dec_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
    decoder_states = [state_h, state_c]
    
    decoder_dense = Dense(num_dec_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # compiles the model
    model.compile(optimizer=optimizer, loss=loss)
    
    # defines the encoder and decoder models for inference
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def deep_lstm(num_enc_tokens, num_dec_tokens, latent_dim=128, 
            optimizer='rmsprop', loss='categorical_crossentropy'):
    """
    tutorial: https://keras.io/examples/nlp/bidirectional_lstm_imdb/
    """
    # ENCODER
    encoder_inputs = Input(shape=(None, num_enc_tokens))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True))
    encoder_outputs, state_h_forward, state_c_forward, state_h_back, state_c_back = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h_forward, state_c_forward, state_h_back, state_c_back]
    
    # DECODER
    decoder_inputs = Input(shape=(None, num_dec_tokens))
    decoder = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    decoder_outputs, _, _, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_dec_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # compiles the model
    model.compile(optimizer=optimizer, loss=loss)
    
    # defines the encoder and decoder models for inference
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h_forward = Input(shape=(latent_dim,))
    decoder_state_input_c_forward = Input(shape=(latent_dim,))
    decoder_state_input_h_back = Input(shape=(latent_dim,))
    decoder_state_input_c_back = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h_forward, decoder_state_input_c_forward, decoder_state_input_h_back, decoder_state_input_c_back]
    decoder_outputs, state_h_forward, state_c_forward, state_h_back, state_c_back = decoder(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_forward, state_c_forward, state_h_back, state_c_back]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
