import torch
import torch.utils.data
from torchvision.utils import save_image
from torch import optim
import numpy as np
from models.GreyUTKFaceVAE import Model
import os
import shutil
from processed_data import GreyUTKFace
from utils import device
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.switch_backend('agg')

def generate_face_gif():
    test_dataset = GreyUTKFace.Dataset(train=False, sample=False)

    model = Model(device)
    model.load()

    ## Get Latent Point Average For Each Age

    latent_ages = []

    latent_ages.append(np.zeros(model.latent_space))
    model.to(device)
    print("Getting latent points for average face of each age")
    for age in tqdm(range(test_dataset.min_age, test_dataset.max_age + 1)):
        items = test_dataset.get_faces_of_age(age)
        faces = list(map(lambda item: item[0].numpy(), items))
        if len(faces) == 0:
            latent_ages.append(np.zeros(model.latent_space))
            continue
        faces = np.array(faces)
        face_imgs = torch.from_numpy(faces).to(device)
        means, _ = model.encode(face_imgs)
        latent_ages.append(np.average(means.detach().cpu().numpy(), axis=0))

    ## Generate Many Steps between Ages

    steps_per_age = 4
    smooth_latent_ages = []
    for i in range(len(latent_ages) - 1):
        age = latent_ages[i]
        next_age = latent_ages[i+1]

        smooth_latent_ages.append(age.copy())

        if steps_per_age <= 1:
            continue

        step_vector = (next_age - age) / steps_per_age
        curr = age
        for i in range(steps_per_age - 1):
            curr += step_vector
            smooth_latent_ages.append(curr.copy())   

    tensor_latent_ages = torch.from_numpy(np.array(smooth_latent_ages)).to(device).float()

    face_ages = model.decode(tensor_latent_ages).cpu()

    # Clear old data
    try:
        shutil.rmtree(model.dir + 'results/gif/')
    except:
        pass
    try:
        os.makedirs(model.dir + 'results/gif/')
    except:
        pass
    
    print("Making GIF images")
    for i in tqdm(range(len(tensor_latent_ages))):
        plt.figure()
        plt.imshow(face_ages[i].view(GreyUTKFace.Dataset.width, GreyUTKFace.Dataset.height).detach().numpy(), cmap='gray')
        plt.title('Age: {}'.format(round(i / steps_per_age, 2)))
        plt.savefig(model._model_path + 'results/gif/{}.png'.format(i))
        plt.close()

    ## Make a GIF

    import glob
    import os

    gif_path = model._model_path + 'results/age.gif'
    img_list_file_path = model._model_path + 'results/gif/image_list.txt'
    file_list = glob.glob(model._model_path + 'results/gif/*.png') # Get all the pngs in the current directory
    list.sort(file_list, key=lambda x: int(x.split('.png')[0].split('/')[-1])) # Sort the images by #, this may need to be tweaked for your use case

    print("Compiling images into GIF...")
    with open(img_list_file_path, 'w') as file:
        for item in file_list:
            file.write("%s\n" % item)
    command = 'convert @{} {}'.format(img_list_file_path, gif_path)
    os.system(command)
    print("Done")