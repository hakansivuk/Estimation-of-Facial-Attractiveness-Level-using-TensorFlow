
import tensorflow as tf
import numpy as np

from training.data import get_dataset
from training.loss import Loss
from training.custom_sched import set_lr

from utils.arguments import Arguments
from utils.logger import Logging
from utils.stats import Stats


#Set GPU if it is available
physical_devices = tf.config.list_physical_devices('GPU')
if (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from training.models.vanilla import VanillaModel

tf.random.set_seed(100)

#Arguments
args = Arguments()
args = args.get_training_args()


#Set Logger
logger = Logging(args.experiment_name)
logger.log_args(args)

#Set Stats
stats = Stats(args.experiment_name)


batch_size = args.batch_size
SHUFFLE_BUFFER_SIZE = 100  # Used to shuffle the data

#Load Dataset
train_dataset, train_len = get_dataset(mode="train")
train_loaded_size = np.ceil(train_len / batch_size)
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset, val_len = get_dataset(mode="valid")
val_loaded_size = np.ceil(val_len / batch_size)
val_dataset = val_dataset.batch(batch_size)

test_dataset, test_len = get_dataset(mode="test")
test_loaded_size = np.ceil(test_len / batch_size)
test_dataset = test_dataset.batch(batch_size)



#Create Model
if (args.network_type == "vanilla"):
    model = VanillaModel(args.bn, args.dropout,args.drop_rate,args.vanilla_conv_count,args.init, args.kernel_l2, args.bias_l2)

#Loss Function
lossfunc = Loss(loss_function=args.loss_function)

#Learning rate scheduler
if (args.lr_sched == "none" or args.lr_sched == "custom"):
    lr = args.lr
elif (args.lr_sched == "exp"):
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=args.exp_decay_steps,
        decay_rate=args.exp_decay_rate)
elif (args.lr_sched == "step"):
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [5000], [args.lr, 1e-4])


#Optimizer
if(args.optim_name == "sgd"):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=args.sgd_momentum)
elif args.optim_name == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=args.adam_beta1, beta_2=args.adam_beta2)
elif args.optim_name == "adagrad":
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
elif args.optim_name == "rmsprop":
    optimizer =tf.keras.optimizers.RMSprop(learning_rate=lr)


patience = args.patience
last_val_mae = 10000 #A very high value
no_imp_count = 0

lowest = 10
for epoch in range(1, args.num_epochs +1):
    train_loss_list = np.zeros(shape=(int(train_loaded_size),))
    #val_loss_list = np.zeros(shape=(int(val_loaded_size),))
    
    #last_val_loss = 0
    sum_mae_train = 0
    sum_mae_val = 0

    for batch_idx, (images, labels) in enumerate(train_dataset):
        #print("Batch no:", batch_idx)
        #Forward pass, which creates Tensorflow Graph
        with tf.GradientTape() as tape:
            preds = model(images, training = True)
            #pred_np = preds.numpy()
            loss = lossfunc(labels, preds)

        grads =  tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_list[batch_idx] = loss.numpy()

        #Mean Absolute Error in Train Set
        preds = np.rint(preds.numpy())   
        sum_mae_train = sum_mae_train + np.sum( np.absolute(preds-labels.numpy()))
    
    #Calculate Mean Absolute Error in Validation Set
    for batch_idx, (images, labels) in enumerate(val_dataset):
        preds_eval = model(images, training = False)
        preds_eval = preds_eval.numpy()
        preds_eval = np.rint(preds_eval)   
        sum_mae_val = sum_mae_val + np.sum( np.absolute(preds_eval-labels.numpy()))

    average_train_loss = train_loss_list.mean()
    average_train_mae = sum_mae_train/train_len
    average_val_mae = sum_mae_val/val_len
    logger.log(f"Epoch: [{epoch}/{args.num_epochs}], Epoch Train Loss: {average_train_loss}, Train MAE: {average_train_mae}, Valid MAE: {average_val_mae}")
    stats.update_train_loss_stats(y_axis_value=average_train_loss, x_axis_value=epoch)
    stats.update_train_mae_stats(y_axis_value=average_train_mae, x_axis_value=epoch)
    stats.update_val_mae_stats(y_axis_value=average_val_mae, x_axis_value=epoch)
    #stats.update_stats(average_train_mae, average_val_mae, epoch)
    if (args.lr_sched == "custom"):
        set_lr(epoch = epoch, optimizer=optimizer)

    #Early stopping
    if (average_val_mae < lowest):
        model.save_weights(f'checkpoints/{args.experiment_name}/model')
        logger.log("Model Saved!")
        lowest = average_val_mae
    #Early Stopping
    # if epoch > args.patience_start and average_val_mae > last_val_mae:
    #     if no_imp_count == patience - 1:
    #         logger.log(f"No validation loss improvement for {patience} epochs!")
    #         break
    #     else:
    #         no_imp_count += 1
    # else:
    #     no_imp_count = 0

    last_val_mae = average_val_mae
    
    # for batch_idx, (images, labels) in enumerate(val_dataset):
    #     preds = model(images, training=False)
    #     loss = lossfunc(labels, preds)
    #     val_loss_list[batch_idx] = loss.numpy()
    
    # average_val_loss = val_loss_list.mean()
    # logger.log(f"Epoch: [{epoch}/{args.num_epochs}], Epoch Val Loss: {average_val_loss}")
    # stats.update_val_loss_stats(y_axis_value=average_val_loss, x_axis_value=epoch)

    # if epoch > args.patience_start and average_val_loss > last_val_loss:
    #     if no_imp_count == patience - 1:
    #         logger.log(f"No validation loss improvement for {patience} epochs!")
    #         break
    #     else:
    #         no_imp_count += 1
    # else:
    #     no_imp_count = 0

    # last_val_loss = average_val_loss

#Log the model for future reference
logger.log_model(model)

#Save the model
# model.save_weights(f'checkpoints/{args.experiment_name}/model')
# logger.log("Model Saved!")

#Load the best performed model
model = VanillaModel(args.bn, args.dropout,args.drop_rate,args.vanilla_conv_count,args.init, args.kernel_l2, args.bias_l2)
model.load_weights(f'checkpoints/{args.experiment_name}/model')

# Test the model on test set
# images, labels = get_dataset(mode="test", dataset=False)
# preds = model(images, training=False)
# mae_score = tf.keras.metrics.mean_absolute_error(labels, preds)
# logger.log(f"Mean absolute error on test set is {mae_score}")

sum_mae = 0
for batch_idx, (images, labels) in enumerate(test_dataset):
        #print("Batch no:", batch_idx)
        preds_test = model(images, training = False)
        preds_test = preds_test.numpy()
        preds_test = np.rint(preds_test)   
        sum_mae = sum_mae + np.sum( np.absolute(preds_test-labels.numpy()))

logger.log(f"Test MAE: {sum_mae/test_len}")
