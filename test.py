import tensorflow as tf
import numpy as np

from training.models.vanilla import VanillaModel
from utils.arguments import Arguments
from utils.logger import Logging

from training.data import get_dataset
import pandas as pd

args = Arguments()
args = args.get_training_args()

#Set Logger
logger = Logging(args.experiment_name)
logger.log_args(args)

#Set GPU if it is available
physical_devices = tf.config.list_physical_devices('GPU')
if (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



model = VanillaModel(args.bn, args.dropout,args.drop_rate,args.vanilla_conv_count,args.init, args.kernel_l2, args.bias_l2)
model.load_weights(f'checkpoints/combined/Nov20-02:21/model')
batch_size = 1
gt = []
pred = []
images, labels, names = get_dataset(mode="test", dataset=False)
for i in range(len(names)):
        #print("Batch no:", batch_idx)
        #print("Image Name: ", names[i])
        preds_test = model(np.expand_dims(images[i], axis=0), training = False)
        preds_test = preds_test.numpy()
        preds_test = np.rint(preds_test)
        if(np.abs(preds_test[0] - labels[i]) >= 2 and labels[i]==1):
            print("Image Name: ", names[i])
            print("Test Predictions: ", preds_test)
            print("Ground Truth: ", labels[i])
        pred.append(int(preds_test[0]))
        
        gt.append(labels[i])


df = pd.DataFrame({"gt":gt, "pred":pred})
df["error"] = np.abs((df["gt"]-df["pred"]))

cnt_df = df.groupby(["pred"])["error"].mean().reset_index()
print(cnt_df)

cnt_df = df.groupby(["gt"])["error"].mean().reset_index().rename(columns={"gt": "Ground Truth", "error": "MAE"})
print(cnt_df)