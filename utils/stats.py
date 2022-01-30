import tensorflow as tf

class Stats():
    def __init__(self, experiment_name):
        train_log_dir = f'results/{experiment_name}/stats'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def update_train_loss_stats(self,y_axis_value, x_axis_value):

        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_loss', y_axis_value, step=x_axis_value)

    def update_val_loss_stats(self,y_axis_value, x_axis_value):

        with self.train_summary_writer.as_default():
            tf.summary.scalar('val_loss', y_axis_value, step=x_axis_value)

    def update_train_mae_stats(self,y_axis_value, x_axis_value):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_mae', y_axis_value, step=x_axis_value)

    def update_val_mae_stats(self,y_axis_value, x_axis_value):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('val_mae', y_axis_value, step=x_axis_value)

    def update_stats(self, train_mae, val_mae, epoch):

        self.train_summary_writer.add_scalars('mae', {'train': train_mae, 'val': val_mae}, step=epoch)