import tensorflow as tf

class Callback(tf.keras.callbacks.Callback):
    
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def on_epoch_end(self,batch, logs=None):    
        print("\nEpochs Ending Custom Callbacks:Intiation Predicting on test set:::")
        logs['from custom callback']=self.model.evaluate(self.x,self.y)[0]
def scheduler(epoch, lr):

    if epoch %2 == 0 :
        print("Epoch:",epoch,"Learning Rate:",lr)
        return lr
    else:
        print("Epoch:",epoch,"Learning Rate:", lr * tf.math.exp(-0.1))
        lr_updated= lr * tf.math.exp(-0.1)
        return  lr_updated