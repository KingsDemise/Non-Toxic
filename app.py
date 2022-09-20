import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
read=pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv'))
from tensorflow.keras.layers import TextVectorization
com=read['comment_text']
val=read[read.columns[2:]].values
maxwords=50000
map=TextVectorization(max_tokens=maxwords,output_sequence_length=1000,output_mode='int')
map.adapt(com.values)
map('I am a bad guy, duh')[:6]
mapped_txt=map(com.values)
mapped_txt
ds=tf.data.Dataset.from_tensor_slices((mapped_txt,val))
ds=ds.cache()
ds=ds.shuffle(160000)
ds=ds.batch(16)
ds=ds.prefetch(8)
ds.as_numpy_iterator().next()
train=ds.take(int(len(ds)*.7))
valid=ds.skip(int(len(ds)*.7)).take(int(len(ds)*.2))
test=ds.skip(int(len(ds)*.9)).take(int(len(ds)*.1))
model=Sequential() 
model.add(Embedding(maxwords+1,32))
model.add(Bidirectional(LSTM(32,activation='tanh')))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(6,activation='sigmoid'))
model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()
training=model.fit(train,epochs=10,validation_data=valid)
ip=map('Your Mom Gay')
ip
model.predict(np.expand_dims(ip,0))
pre=Precision()
rec=Recall()
cat=CategoricalAccuracy()
for batch in test.as_numpy_iterator(): 
    x_true,y_true=batch
    yhat=model.predict(x_true)
    y_true=y_true.flatten()
    yhat=yhat.flatten()
    pre.update_state(y_true,yhat)
    rec.update_state(y_true,yhat)
    cat.update_state(y_true,yhat)
print(f'Precision: {pre.result().numpy()}, Recall:{rec.result().numpy()}, Accuracy:{cat.result().numpy()}')
import gradio as gr
model.save('toxicity.h5')
model = tf.keras.models.load_model('toxicity.h5')
ip=map('You are a bad human being, I hope you rot in hell.')
res=model.predict(np.expand_dims(ip,0))
res
def fun(Comment):
    map_com=map([Comment])
    results=model.predict(map_com)
    s=''
    for i,j in enumerate(read.columns[2:]):
        s+='{}: {}\n'.format(j,results[0][i]>0.5)
    return s
interface=gr.Interface(fn=fun,inputs=gr.components.Textbox(lines=2,placeholder='Comment Toxicity'),outputs='text')
interface.launch(share=True)    
                                                                                                        
                             
