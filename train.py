from models.UTKFace.age_classifier import AgeClassifier
from utils import device

# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--load', action='store_true', default=False, help='Skip Training')
# args = parser.parse_args()

model = AgeClassifier()
model = model.to(device)
model.train_model(num_epochs=1000, model_output_path="model.pt", log_interval=1)
