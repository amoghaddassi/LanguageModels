# Learning language models

Goal: want to get a solid understanding of the current state of the art in ML, and also pick up the fundamental ideas and skills along the way.

Plan: start by exploring popular models (GPT-3, MuZero), which should hopefully expose gaps in your understanding. Develop small projects as you go. Develop a big project at the end (maybe finally do that drug discovery or EEG model).

Possible project idea for the language model: train a transformer on all office scripts. Have the model generate a totally novel script.


## Code

As I go through the papers, I think it's wise that I code these networks in Keras and train them on small text sets to make sure that I have an understanding. Will do everything in Keras b/c that's easy. For testing, will use jupyter notebooks. When I'm ready to do something larger/more final, will do it in dedicated python files.

Generally, I am NOT opposed to just copying code from other places on the internet. Why struggle with syntax when I can just see what works and internalize the solution.

### RNN encoder-decoder

Want to start here because, while it is old tech, it will give me an appreciation for the starting point of neural machine translation. Also, it doesn't seem that hard to code so a nice warmup for the more complex projects I want to work on.

Steps:
- Find a corpus of translation data. Sequences can't be too long, since this architecture is prone to the vanishing gradient problem.
- Write the encoder: just an RNN with a fixed length hidden state.
- Write the decoder: still not totally sure how this is implemented for seq2seq models.
- Put them together and train.

#### Data

Got some english to french aligned translations from the Canadian government. Let's unpack this and see what we're working with.

Ok got all the files in a directory that I can work with. Now let's bring these into python. Got all the file pairs into python without a hitch (and split them into train, test, validation). Now need to figure out how to parse the text from these files. They get errors when I try file.read(), so I assume I need to use a better encoding. Sweet, just needed to change the encoding from UTF-8. This all looks good to me, what's next?

Note: I should probably send these text samples through the network 1 line at a time (they're aligned anyway, so why not take advantage of it). This will massively reduce the vanishing gradient problem when I'm dealing with vanilla (no attention) networks. Ok done, the datasets now just contain line (rather than full file) pairs.

Another potential issue might be grammar outside the typical alphabet, but will ignore this for now. In any case, will need to work out the number of tokens so that I can do one hot encoding for the network. Hmm ... doing this in the naive way is just way too slow, need to think of a clever way to do/guess this. Maybe use the file.read() encoding as a reference.

Ok need to go about this another way. Better than counting all the unique characters, I'll just predefine an alphabet for each language (only lower case) and filter all input text through that. Effectively this will remove all the punctuation and special characters, but I think that's for the best of the network. Will take a while to run this once, but then I just save them as .npy files.

Data is looking clean! Let's get these saved in .npy files and finish this. Done. Let's get to the good stuff (though that was kind of fun if I'm being honest).

Had some more steps to get the data really ready to send into a model. I had to one_hot encode it and pad everything in numpy 3d arrays, that should be ready to go to send into the keras model. Saved these instead of the raw text version.

#### Keras seq2seq tutorial

Perfect starting point! Let's just follow this.

Copy and pasted all the model specification code. Looking at a summary of it right now. Realizing I don't know as much Keras as I thought. Before I train this, let's make sure I have a working understanding of what all this model code actually does.

Ok at the very least understand how data flows. Still need to look into this decoder_input field. Makes sense that it's required, but still don't fully get it. Anyway, training now, should be about an hour.

Now I need to setup inference. Loop is basically just taking greedy choice from softmax each time, then feeding that back for next letter. Pseduocode:

translate(src):
    target = encode(src)
    while target not done:
        c = decode(target)
        target += c
    return target
    
Shit just realized I messed up a bit. I should have included the new line character in the training points so that the model would also have some way to indicate that it thinks the target should be finished. It doesn't have that so outputs will go on arbitrarily long. Going to cop out and just have a manual stopping condition, but going to fix this bug so that I can retrain later on. Also added a start character so that the offset would be less awkward for the target seq.

Getting bugs...what's new. Everything I feed this inference system, it just gives me back "ezezezezeze..." This is a pretty glaring bug, and I think the issue must be with something after the model was trained (how else could it acheive such high training and validation scores, but then totally whiff training examples during inference).

Debugging:
- Getting error with load_model. Quick attempt to fix this - retrain the model with 1 epoch, save with the .h5 extension. This is how I've seen it done everywhere, so maybe this will magically fix my problems.
    - Nice! Just changing the file extension totally solved my load_model problem. Let's see if inference works now.
- Hmmmm now getting "egegegege...". So this looks like a real bug. Let's strap in for this.
- Trying to run the model manually (so I don't have to think about splitting the encoder and decoder), but getting an error that it could not compute the output. What gives?
- Ok got this to run with model.predict (why doesn't model(...) work?) and am getting output that is not just oscillating characters. It probably doesn't make sense, but then why I am getting this output from the encoder decoder. Well when I do model.predict it does get the full french sentence. Could I make the model predict only 1 character?
- Just parsed the output. Seems like model.predict works fine (ie when it receives the output sentence already). This makes sense since we're just giving it the output. But then is the way we're training the model different from how we're evaluating it during inference? How to align this?
- Ah I think I know the problem: the model is just learning to send a delayed version of the output (explains why the categorical cross entropy is low). Also explains why we get silly output when doing full inference. How to fix this?
    - Could be a problem with how I shift my target dec data from my input dec data. Can look at reference code for this.
    - Less likely (since the ref code worked): need a cost function that penalizes this offset by 1 error. Forget exactly how cross entropy works on sequences, but would assume that it hasn't been penalizing this.
- Think shifting target wrong has highest chance of causing bug so far. Let's fix this:
    - Ok sweet, just copied the code that the reference used. Training again, and getting different (but promising) loss values. Let's see where this goes.
    - Good looks! Model kind of works now.
    
Let's mess around with this for a bit to get a feel for how the model is doing.
- It's always spitting out a version of the same sentence. At least this means it learned a few words. Problem is, that's all its using. Is this a problem with the model being too small? Might have to copy params from the example code.
- Ok, way I see it is that I have an architecture and code base that works, but I'm just getting some params wrong. What happens if I copy the example?
- Doesn't look like my params are even that off. Let's just train it for longer to see if that does the trick. Past that, it might be an issue with my data (:/). 

Ugh, retrained and still getting the same output. What's going wrong??
- Maybe it's a data thing. Hard to believe since the data looks pretty good to my eyes. Trying the data used in the example might be a good move though.
- More I look at outputs of this network, more I'm convinced this is now a problem with it not learning. It can do words, which all things considered is validation of the code base. Know the architecture is good for other data. Cood be hparams. Could be bad data. Lets tweak things first before drastically changing anything.
- Note: the reference code includes all characters from the data. I do filtering instead. Prefer my way b/c it is more resource efficient (and I'd think easier to learn). Difference none the less.
- Going to continue training the 6 epoch one and monitor the output change every few epochs. If it gets better, I keep going.
    - Not getting much change, even when I train on the test and valid sets.
    
Progress:

English: 	private bill to amend act of incorporationpresentation of petition 

French (acutal): 	projet de loi dintért privé modifiant la loi constituant en personne morale présentation dune pétition 

French (model + feedback): jaone  de laisde  e e daoseedenininnt de pansdont io  nt entdrrt nte lensta paosant uion de   larie on d                                                                                                                     

French (model): je te le pour de proje de

How many words does the model know? Let's count.

Usage in first 50 training examples:

{'le': 55,
 'sont': 24,
 'de': 93,
 'la': 60,
 'projet': 31,
 'comme': 17,
 'ce': 5,
 'te': 17,
 'pour': 8,
 'proje': 22,
 'proj': 4,
 'en': 3,
 'sénateur': 9,
 'je': 12,
 'pro': 3,
 'ent': 1,
 'com': 2}

Getting better. Needs to be deeper!!

How to be deeper? Struggling with API.

Might be time to bring in attention. These are long sentences, just a RNN won't do the job.

Throw compute at it? Always AWS. But should locally optimize the model much more before that.

Can also jump on some free GPUs/TPUs through Colab --> good first option.

Bidirectional LSTM! 

Things to do:
- Refactor the architecture to a 2 layer bidirectional LSTM (should help with the limited vocab problem). Validate that this works locally.
- Move to Colab and train the full model.
- Evaluate model.

Model: encoder and decoder are both 256 node LSTM layers. Input is reversed. Trained on ~11k examples. 10 epochs. Took ~3.5 hrs to train on macbook.

Output:



#### Deep bidirectional LSTM

This seems like the peak the model can be without adding an attention mechanism. Before I code the model itself, I want to write some python library code that will help with loading data and training.

Ok, very happy with how I packaged everything. Now I just have to make a few function calls from the notebook, and everything works like a charm. Have to change things around a little bit to make sure that doing inference afterwards is as easy as possible (want to return a seperate encoder and decoder with the model definition).

That wasn't so bad. Just have my model definition code returning the full model, encoder model, and decoder model. I can then have a general inference/translate function that just accepts an encoder and decoder model. Have a basic lstm training while I develop the new model now. Might need to also save the enc and dec (as opposed to only saving the full model right now), but don't have to worry about that right now.

Just rewrote the bidirectional code and have it compiling with only one layer per encoder and decoder. Let's sanity check that this works, then move on to a deeper model.

Getting a shape problem with my target. Have to google around for this one. I got it! Just needed to set return_sequences to True in the decoder LSTM layer (this means we'll get back things that have shape (batch_size, sent_length, alphabet_length)). Did a few training loops, and it seems to be learning. Let's do a commit to make sure I don't lose this place.

Let's make it deep!

Bug when evaluating:
- Says I have too many inputs, expected 3, everywhere else says 5 inputs (what i'd expect)
- Can always rebuild models using the get_layer method.

TOO MANY BUGS. Fix later.

Main bug I see right now is that the bi model returns blank sentences only. Don't think this is possible given how good the cross entropy scores were (10^-6), though this could be wrong. Is there a problem with the translate function somewhere?

Redoing this with smaller data to see if there's any other problems. Same problem as before where it only outputs spaces. Hmmm ... is this an issue with the translate function then? Seems that the problem is that it's not learning a context dependent distribution over the letters (just a guess though). This happens after any amount of training. Let's scope what happens with a feedback prediction.

Just gives me 'eeeeeee'. Not sure how to put this all together yet -- other than the model probably needs to be deeper, larger, and train for longer.

Made it deep! And it trains on small samples. Let's do a few more sanity checks and let it loose.

Interesting results after training for about 5hrs. Seems like the bidirectional feature actually hurts the system for longer sentences (need to ponder/create intuition for why this may be true). Makes sense that just the reverse would perform the best in that case. Maybe it's time to add the attention mechanism? Let's do a commit first.

### RNN with attention




#### API notes

encoder_input --> encoder_lstm  --> { --> decoder_lstm --> dense
                  decoder_input --> {
                  

Output of dense is compared against decoder_target = offset(decoder_input).

Two ways to define a model: Model() and Sequential().
- Model: map dummy inputs to outputs with usual python function calling syntax, then compiles the model with all layers.
- Sequential: define the model just in terms of a list of layers.
These ways differ in how the model is defined. After, they share:
- Model.compile(optimizer, loss)
- Model.fit(x, y, batch_size, epochs)

## Papers

### RNN encoder-decoder

This is the starting point architecture for NMT. Should understand it reasonably well (maybe even code it).

The encoder is a pretty straightforward RNN:
- h_t = f(x_t, h_{t-1}), where f is a RNN
- c = q({h_t}) = h_T (usually)
    - This output function seems to be where you'd add the attention mechanism, but for simple networks just take the context to be the last hidden state. Shouldn't be a big deal if you're only dealing with short sequences.
    
The decoder is a little less straightforward:
- Will model the probability dist for each of the output chars in the target sequence. Probs are conditioned on all previous output chars and the context from the encoder RNN.

Equations:
- s_t = sigmoid(Ux_t + Ws_{t-1} + b)
    - typical RNN architecture with sigmoid activation. Nothing fancy here.
- y_t = softmax(Vs_T + c) <-- don't quite get this, what is c?

Youtube videos, let's get basic:
- Enc/dec structure fits a lot of common models: e.g. a CNN's feature extractor layers are all encoders into some compressed image state. The decoder is just a softmax layer that then predicts a class.
- Actually very interesting: does this abstraction fit all neural net models? 

Original seq2seq paper:
- Score increased when encoder (but not decoder) inputs had a reversed order.
    - Makes sense: harder to get start than end, and vanishing gradient means start will have less info from hidden state.
- Used deep LSTMs. Found they performed better than shallow.
- Trained models with a large vocab (160k encoder words, 80k decoder words). My model does things at the letter level; how does this change the problem?

### Align and Translate (Attention in RNNs)

This paper is the key prerequesite to understanding the attention part of the transformer. I want a thorough understanding of this model, so let's start here.

#### Top

Abstract: starting point is that most neural machine translaters (as opposed to the Markov style statistical methods) are encoders and decoders with a fixed length hidden state. Hypothesis is that this fixed length state is a bottleneck. Use an attention mechanism to search for parts of the source sentence that may be relevant to the target. Find that it does well for translation tasks.

#### Attention detour

Feels like I need to appreciate what an attention mechanism is before I dive into the transformer architecture. Don't want to read a paper on this (mostly b/c it doesn't seem like there is one seminal work for this), so I'll just google around and take notes.

Seq2Seq models seems like the start: 
input sequence in source language --> encoder --> RNN hidden layer --> decoder --> output sequence in target language

Reminds me a lot of the autoencoders that I saw when doing the drug discovery stuff. Basic idea is that use an RNN to generate an abstract representation of the sequence in a high dim space, and then use a decoder to translate that to an output language. Think of the encoder and decoder as feed forward nets.
- This actually is a misunderstanding of how seq2seq models work. There are 2 RNNs: an encoder and a decoder. The output of the encoder is fed into the decoder. Not exactly sure how this works, but does seems a bit different (at least on the surface, though effectively these models might be doing the same thing) from the autoencoders I looked at.

Vanishing gradient problem: for long sequences, important information in the start of the seq will be lost in the final hidden state. 
Solution: align and translate. Use a context vector that preserves info from all points in the sequence and aligns them with the target sequence (what does this actually mean?).

Paper on Luong attention mechanisms:
- Global and local attention mechanisms: whether the context vector considers all source positions, or just a subset of them.
- Basic idea of both is that the hidden state is combined with the context vector to produce an "attention vector", which can be given to the decoder. Only difference between the two models is how the context vector is derived.
- Formally, there is a concatenation layer that combines the hidden state with the context vector, and a softmax layer that outputs the distribution over target letters given the output of the concatenation layer.

Align and translate seems to be the first model that used attention with RNNs for machine translation. Questions and comments:
- What is a bidirectional RNN/encoder decoder model? This seems to be the foundation of the attention mechanism. Assumes it's just a pipeline where an encoder RNN feeds into a decoder RNN.
	- Think a bidirectional RNN is something with forward and backward sequence connections. How does training/running work? Seems like you'd have circular constraints. 

### Transformers (Attention is all you need)

This seems like the root of the GPT idea, so might as well start here.

#### Top

Abstract: Beat state of the art language translation performance using a simple feed forward net with attention mechanisms. Trains much faster because it can be parallelized.

There is a fundamental performance limit to RNNs, which is that they must process sequential data in order. i.e. the hidden state at time t depends on the hidden state at time t-1 and the input at time t. This makes parallelizing hard, so runs slow for longer sequences. (Not mentioned, but) also faces the vanishing gradient problem (maybe this is solved by LSTMs), where long term dependencies are forgotten.

Attention mechanisms are used to recover long term dependencies in RNNs. Novel idea: use the attention mechanism in a non-recurrent architecture?

A language model is just a distribution over characters given some string. Can easily make this generate text by always feeding it its own predictions. 

How does this compare to info theory style markov models?
- Markov model only knows statistical properties of the language, but has no learning mechanism (?)
- Can't look too far back for a Markov model. For a traditional RNN, this is analogous to the vanishing gradient problem. How does the transformer avoid them?
	- Can also describe this as: we need a language model to deal with long term dependencies.

### GPT-3

#### Top

Absract: seems like the crux of the model is that they just scaled up the GPT-2 network, and they were able to generalize well to specific domains with limited context. Scale solving problems seems like a theme across deep learning; what does that say about the field?

Questions: 
- What is the architecture of the usual language model? 
- How did they scale that up for GPT-3?

#### Figures

##### 1.1: Language model meta-learning
- Takeaway: the model is exposed to sequences from different contexts during pre-training. Want to give it pattern recognition abilities so that it can quickly pick up on the context at inference.
- Method: outer loop over sequences from different contexts, inner loop that learns those sequences.
- Questions:
	- Is it explicitly learning a mapping from sequences --> contexts, or is that implicit in the weights?
	- Is there some training period after pre-training? i.e. what more does it need to learn before inference?

##### 1.2: Larger models more efficient with in-context information
- Takeaway: 




