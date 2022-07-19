import tensorflow as tf
import tensorflow.keras as keras
    
class DLinear(keras.models.Model):
    """
    DLinear model as outlined in literature:
    https://arxiv.org/pdf/2205.13504.pdf
    
    Input and output data is expected in (batch, timesteps, features) format.
    """
    def __init__(self, output_shape, separate_features=False, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_steps = output_shape[0]
        self.output_features = output_shape[1]
        self.separate_features = separate_features
        self.kernel_initializer = "he_normal"
        
        
    def build(self, input_shape):
        """
        Build function to create necessary layers.
        """
        self.built_input_shape = input_shape
        
        if self.separate_features:
            self.trend_dense = []
            self.residual_dense = []
            for feature in range(self.output_features):
                self.trend_dense.append(keras.layers.Dense(self.output_steps,
                                                           kernel_initializer=self.kernel_initializer,
                                                          name="trend_decoder_feature_"+str(feature)))
                self.residual_dense.append(keras.layers.Dense(self.output_steps,
                                                               kernel_initializer=self.kernel_initializer,
                                                              name="residual_decoder_feature_"+str(feature)))   
        else:
            self.trend_dense = keras.layers.Dense(self.output_steps*self.output_features, 
                                                  kernel_initializer=self.kernel_initializer,
                                                 name="trend_recomposer")
            self.residual_dense = keras.layers.Dense(self.output_steps*self.output_features, 
                                                     kernel_initializer=self.kernel_initializer,
                                                    name="residual_recomposer")
        
    def call(self, inputs):
        """
        I provide 2 settings to DLinear, as defined in literature.
        
        DLinear-S: separate_features = False
        Uses all input features to directly estimate output features, using 2 linear layers.
        
        DLinear-I: separate_features = True
        Uses all input features to directly estimate output features, using 2 linear layers
        PER OUTPUT CHANNEL.
        Theoretically better if scaling of output variables differ.
        """
        trend = keras.layers.AveragePooling1D(pool_size=self.kernel_size,
                                              strides=1,
                                              padding="same",
                                              name="trend_decomposer")(inputs)
        
        residual = keras.layers.Subtract(name="residual_decomposer")([inputs, trend])
        
        if self.separate_features:
            paths = []

            for feature in range(self.output_features):
                trend_sliced = keras.layers.Lambda(lambda x: x[:, :, feature],
                                                  name="trend_slicer_feature_"+str(feature))(trend)
                trend_sliced = self.trend_dense[feature](trend_sliced)
                trend_sliced = tf.keras.layers.Reshape((self.output_steps, 1),
                                                      name="reshape_trend_feature_"+str(feature))(trend_sliced)
                
                residual_sliced = keras.layers.Lambda(lambda x: x[:, :, feature],
                                                      name="residuals_slicer_feature_"+str(feature))(residual)
                residual_sliced = self.residual_dense[feature](residual_sliced)
                residual_sliced = tf.keras.layers.Reshape((self.output_steps, 1),
                                                          name="reshape_residual_feature_"+str(feature))(residual_sliced)
                
                path = keras.layers.Add(name="recomposer_feature_"+str(feature))([trend_sliced, residual_sliced])
                
                paths.append(path)
                
            reshape = keras.layers.Concatenate(axis=2,
                                              name="output_recomposer")(paths)
        else:
            flat_residual = keras.layers.Flatten()(residual)
            flat_trend = keras.layers.Flatten()(trend)

            residual = self.residual_dense(flat_residual)
            
            trend = self.trend_dense(flat_trend)

            add = keras.layers.Add(name="recomposer")([residual, trend])

            reshape = keras.layers.Reshape((self.output_steps, self.output_features))(add)
        
        return reshape
    
    def summary(self):
        """
        Override model.summary to allow usage on nested model.
        """
        if self.built:
            self.model().summary()
        else:
            # If we haven't built the model, show the normal error message.
            super().summary()
            
    def model(self):
        """
        Workaround to allow for methods on model to work.
        Model nesting gets janky in tensorflow, apparently.
        
        Use model.model() in place of model.
        
        e.g. tf.keras.utils.plot_model(model.model())
        """
        x = keras.layers.Input(shape=(self.built_input_shape[1:]))
        model = keras.models.Model(inputs=[x],outputs=self.call(x))
        
        return model
