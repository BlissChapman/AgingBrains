import argparse 

from models.GreyUTKFace import AgeClassifier
from models.GreyUTKFace import VAE

from utils import device

# Parse Arguments
parser = argparse.ArgumentParser(description='Train any model in the models directory!')
parser.add_argument('model_type', type=str,
                    help='the model type to train (ex: GreyUTKFace.AgeClassifier)')
parser.add_argument('--sample', action='store_true', default=False, 
                    help='Sample a small set of the data to make it run faster. Useful for debugging')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status (default: 10)')
parser.add_argument('--load', action='store_true', default=False,
                    help='whether or not to load the trained weights of the model')
args = parser.parse_args()

# Construct model
if args.model_type == "GreyUTKFace.AgeClassifier":
    model = AgeClassifier.Model()
elif args.model_type == "GreyUTKFace.VAE":
    model = VAE.Model()
else:
    print("Unknown model type: `{0}`".format(args.model_type))
    exit(1)
    
# Load weights
if args.load:
    model.load()

# Train model
model = model.to(device)
model.train_model(num_epochs=args.epochs, 
                  sample=args.sample,
                  log_interval=args.log_interval)
