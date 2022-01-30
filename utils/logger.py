import logging
import sys
import pathlib

class Logging():
    def __init__(self, experiment_name):

        #Crete log dir if not exist
        path = f"results/{experiment_name}"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


        #Print format: Time - Message
        format='%(asctime)s - %(message)s'

        logging.basicConfig(level=logging.DEBUG, 
                            format=format , 
                            handlers=[
                            logging.FileHandler(f"{path}/out.log"),
                            logging.StreamHandler(sys.stdout)])

    def log(self, message):
        logging.info(message)

    def log_args(self,args):
        message = "PROGRAM ARGUMENTS:\n"

        for arg in vars(args):
             message += f"{arg}: {getattr(args, arg)} \n"
        message += "-----------------------------------"
              
        self.log(message)

    def log_model(self, model):

               
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.log(short_model_summary)