import tensorflow as tf

class Loss(tf.keras.losses.Loss):
    def __init__(self, loss_function):
        super(Loss, self).__init__()
        if loss_function == 'mse':
            self.loss = tf.keras.metrics.mean_squared_error
        elif loss_function == 'mae':
            self.loss = tf.keras.metrics.mean_absolute_error
        elif loss_function == 'rmse':
            self.loss = "rmse"
        elif loss_function == "combined":
            self.loss = "combined"
        elif loss_function == 'mape':
            self.loss = tf.keras.metrics.mean_absolute_percentage_error
        elif loss_function == 'msle':
            self.loss = tf.keras.metrics.mean_squared_logarithmic_error
        elif loss_function == 'poisson':
            self.loss = tf.keras.metrics.poisson
        elif loss_function == 'squared_hinge':
            self.loss = tf.keras.metrics.squared_hinge
        elif loss_function == 'log_cash':
            self.loss = tf.keras.losses.log_cosh
        elif loss_function == 'kl_divergence':
            self.loss = tf.keras.metrics.kl_divergence
        elif loss_function == 'huber':
            self.loss = tf.keras.losses.huber
        elif loss_function == 'cosine_similarity':
            self.loss = tf.keras.losses.cosine_similarity
        elif loss_function == 'hinge':
            self.loss = tf.keras.metrics.hinge
        else:
            raise Exception("No loss function is defined with that name!")
       

    def call(self, labels, preds):
        if (self.loss == "rmse"):
            lossval = tf.sqrt(tf.reduce_mean((labels - preds)**2))
        elif(self.loss == "combined"):
            lossval =0.25* tf.sqrt(tf.reduce_mean((labels - preds)**2)) + 0.75 * tf.reduce_mean((labels - preds)**2)
        else:
            lossval = self.loss(labels, preds)
        return lossval
        

    """
    def get_model_loss(self,labels, preds, variables, loss_function):
        loss = losses.mean_squared_error(labels,preds)
        return loss
        reg_penalty = 0
        if (self.regularizer_weight != 0):
            if(self.regularizer_type == 'l2'):
                reg_penalty = tf.reduce_sum(tf.add_n([ tf.nn.l2_loss(v) for v in variables ]))


        #Add here additional losses.
        loss = mse + (self.regularizer_weight*reg_penalty)
        return loss
    """