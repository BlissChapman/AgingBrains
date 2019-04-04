import argparse 

from models import GreyUTKFaceAgeClassifier
from models import GreyUTKFaceVAE

from utils import device

# Parse Arguments
parser = argparse.ArgumentParser(description='Train any model in the models directory!')
parser.add_argument('model_type', type=str,
                    help='the model type to train (ex: GreyUTKFace.AgeClassifier)')
parser.add_argument('--sample', action='store_true', default=False, 
                    help='Sample a small set of the data to make it run faster. Useful for debugging')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status (default: 10)')
parser.add_argument('--load', action='store_true', default=False,
                    help='whether or not to load the trained weights of the model')
args = parser.parse_args()


# Construct model
if args.model_type == "GreyUTKFaceAgeClassifier":
    model = GreyUTKFaceAgeClassifier.Model(device)
elif args.model_type == "GreyUTKFaceVAE":
    model = GreyUTKFaceVAE.Model(device)
else:
    print("Unknown model type: `{0}`".format(args.model_type))
    exit(1)
    
# Load weights
if args.load:
    model.load()
    print("Loaded a previous save with {} epochs trained".format(model.epochs_trained))
elif model.save_exists():
    answer = input('A saved model already exists, are you sure you want to overwrite it? (y/n)').strip()
    if answer != 'y':
        exit()
    
# Train model
for e in range(1, args.epochs + 1):
    print("Epoch: {}".format(e))
    train_loss = model.train_an_epoch(sample=args.sample)
    print('Train loss: {}'.format(train_loss))
    test_loss = model.test(sample=args.sample)
    print('Test loss: {}'.format(test_loss))    
    model.evaluate()
    
    if e % 5 == 0 and e > 0:
        model.save()
        print("Model saved")
model.save()
print("Model saved")