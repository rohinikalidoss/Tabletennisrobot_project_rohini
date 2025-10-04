import tensorflow as tf
from tensorflow.keras import layers, Model
import keras

@keras.utils.register_keras_serializable()
class BatchNormPINN(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Your friend's winning architecture
        self.dense1 = layers.Dense(1024, activation='relu', name='dense1')
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.dropout1 = layers.Dropout(0.3, name='dropout1')
        
        self.dense2 = layers.Dense(512, activation='relu', name='dense2') 
        self.bn2 = layers.BatchNormalization(name='bn2')
        self.dropout2 = layers.Dropout(0.3, name='dropout2')
        
        self.dense3 = layers.Dense(256, activation='relu', name='dense3')
        self.bn3 = layers.BatchNormalization(name='bn3')
        
        self.dense4 = layers.Dense(128, activation='relu', name='dense4')
        self.bn4 = layers.BatchNormalization(name='bn4')
        
        self.dense5 = layers.Dense(64, activation='relu', name='dense5')
        self.bn5 = layers.BatchNormalization(name='bn5')
        
        self.output_layer = layers.Dense(5, activation='linear', name='output')
    
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)  # Critical: training flag
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        
        x = self.dense4(x)
        x = self.bn4(x, training=training)
        
        x = self.dense5(x)
        x = self.bn5(x, training=training)
        
        return self.output_layer(x)
    
    def get_config(self):
        return super().get_config()
