# Experiments

This will contain a history of all models that I've trained.

## 01: basic_lstm

### Architecture

input_9 (InputLayer)            [(None, None, 29)]   0                                            
__________________________________________________________________________________________________
input_10 (InputLayer)           [(None, None, 34)]   0                                            
__________________________________________________________________________________________________
lstm_8 (LSTM)                   [(None, 256), (None, 292864      input_9[0][0]                    
__________________________________________________________________________________________________

lstm_9 (LSTM)                   [(None, None, 256),  297984      input_10[0][0]                   
                                                                 lstm_8[0][1]                     
                                                                 lstm_8[0][2]                     
__________________________________________________________________________________________________

dense_4 (Dense)                 (None, None, 34)     8738        lstm_9[0][0]                     

Total params: 599,586
Trainable params: 599,586
Non-trainable params: 0

### Training

Training: ~11k examples from train_small.npy
Epochs: 10
Batch: 64
Time: ~3.5 hrs
Final loss: .08

### Output

example: 200
====================
English: 	i believe they are acting if i may say this conservatively 

French (acutal): 	à mon avis elle se montre prudente 

French (model + feedback): j let aues ln e do petdre daésent  d                                                   

French (model): je propes de le projet de loi c port ce que le projet de loi c

====================
example: 201
====================
English: 	however we should give some credit to both the government and the economy for presenting us with this interesting challenge  

French (acutal): 	cependant nous devrions reconnatre que le gouvernement et léconomie ont le mérite de nous avoir mis en présence de cet intéressant défi  

French (model + feedback): ce endant lous ae oisns pesonnaire lue le sruvernement dt degondmie dnt de monite de louv avons santsn crosente de lottentere  ent lecicm

French (model): ce qui en propes de la proposition de la propeste de la proposition de la propeste de la proposition de la proposition de la pro

====================
example: 202
====================
English: 	senator oliver 

French (acutal): 	le sénateur oliver 

French (model + feedback): le sénateur gniner 

French (model): le sénateur graham

====================
example: 203
====================
English: 	the honourable minister has perhaps preempted the minister of finance and given us an economic forecast  

French (acutal): 	peuttre lhonorable ministre atil devancé le ministre des finances et nous atil donné des prévisions économiques  

French (model + feedback): losn re deonorable sénastra d  l pe aite de ponistre de  panances dt dous avtllsenté se  prosiseons dtliomiques 

French (model): lhonorable s natera de projet de loi sur le projet de loi c port ce qui en compte de la proposition de la pr

====================
example: 204
====================
English: 	however my specific question dealt with the issue of reserves in the fund 

French (acutal): 	toutefois ma question portait spécifiquement sur la provision de la caisse 

French (model + feedback): coute ois paidue tion dautent deaciaieue ent lur le projecton de la pons   d                         
French (model): ce qui s pour le projet de loi c port ce qui en compte de la proposition de l


### Notes:

Original paper: https://arxiv.org/pdf/1409.3215.pdf
Tutorial: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

Can learn a limited French vocab (~30 words), but struggles with broader pattern recognition. This is definetely a size issue, but is promising b/c it means that the model is learning something (starts with just knowing the alphabet after all).

## 02: Shallow bidirectional LSTM

### Architecture

English input --> encoder bi-LSTM (128 units) --> decoder bi-LSTM (128 units) --> dense --> French output

### Training

Training: ~11k examples from train_small.npy
Epochs: 10
Batch: 64
Time: ~4hrs
Final loss: 1e-6

### Output

All blank spaces. Seems that it's not able to learn context dependent distributions. Need to make this deeper, larger, and train it more.

### Notes
- Lost the model files for this one. Didn't do anything special so oh well.
- Why would the model have such a low final loss but still not perform well? Maybe b/c loss is calculated w/ feedback.

## 03: deep_lstm

### Architecture:

input_89 (InputLayer)           [(None, None, 29)]   0                                            
__________________________________________________________________________________________________
bidirectional_44 (Bidirectional [(None, None, 256),  161792      input_89[0][0]                   
__________________________________________________________________________________________________

input_90 (InputLayer)           [(None, None, 34)]   0                                            
__________________________________________________________________________________________________

bidirectional_45 (Bidirectional [(None, 256), (None, 394240      bidirectional_44[0][0]           
                                                                 bidirectional_44[0][1]           
                                                                 bidirectional_44[0][2]           
                                                                 bidirectional_44[0][3]           
                                                                 bidirectional_44[0][4]           
__________________________________________________________________________________________________

bidirectional_46 (Bidirectional [(None, None, 256),  166912      input_90[0][0]                   
                                                                 bidirectional_45[0][1]           
                                                                 bidirectional_45[0][2]           
                                                                 bidirectional_45[0][3]           
                                                                 bidirectional_45[0][4]           
__________________________________________________________________________________________________

bidirectional_47 (Bidirectional [(None, None, 256),  394240      bidirectional_46[0][0]           
                                                                 bidirectional_46[0][1]           
                                                                 bidirectional_46[0][2]           
                                                                 bidirectional_46[0][3]           
                                                                 bidirectional_46[0][4]           
__________________________________________________________________________________________________

dense_16 (Dense)                (None, None, 34)     8738        bidirectional_47[0][0]           

Total params: 1,125,922
Trainable params: 1,125,922
Non-trainable params: 0

### Training

Training: ~11k examples from train_small.npy
Epochs: 7
Batch: 64
Time: ~40min/epoch
Final loss: ~1e-6

### Output

example: 181

English: 	in his november   report the auditor general raises the question of employment insurance surplus 

French (acutal): 	dans son rapport du  novembre  le vérificateur général parle de lexcédent de la caisse de lassuranceemploi 

French (model + feedback): dans son rapport du  novembre  le vérificateur général parle de lexcédent de la caisse de lassuranceemploi                                                                                                             

French (model): ptlamômôgôgôgôyôlôôôôôôôôôôôôôôôôôôôôôôôôôôôyryuuuuuuuuuyjyyoylzôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôyr

====================
example: 182

English: 	the ei premium rate is set by the ei commission which is composed by representatives from the employees the employers and the government and must be approved by cabinet on the recommendation of the human resources development and finance ministers 

French (acutal): 	les taux de cotisation à lassuranceemploi sont établis par la commission dassuranceemploi qui regroupe des représentants des employés des employeurs et du gouvernement et ils doivent tre approuvés par le cabinet sur la recommandation du ministre des finances et du ministre du développement des ressources humaines 

French (model + feedback): les taux de cotisation à lassuranceemploi sont établis par la commission dassuranceemploi qui regroupe des représentants des employés des employeurs et du gouvernement et ils doivent tre approuvés par le cabinet sur la recommandation du ministre des finances et du ministre du développement des ressources humaines                   

French (model): ptlmmmmmmmmôgôgôgôyôlôôôôôôôôôôôôôôôôôôôôôôôôôôôôôyryuuuuuuuuuyjyyoylzôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôyryuuuuuuuuuyjyyoylzôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôyryuuuuuuuuuyjyyoylzôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôyryuuuuuuuuuyjyyoylzôôôôôôôôôôôôôôôôôôô


### Notes
- So model+feedback gets it pretty much perfect. Just the model does garbage on its own. Explains why we have such a low training loss. But why would just the model do so poorly? This isn't a problem I had with the basic lstm (still the best I have).