# %% First thing first
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

# %% Define model for passing the images through layers
class NeuralStyle(nn.Module):
    def __init__(self):
        super(NeuralStyle, self).__init__()

        self.chosen_layers = [0, 5, 10, 19, 28]
        self.model = models.vgg19(weights='DEFAULT').features[:29]     # We don't need after 28...

    def forward(self, x):
        activations = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)    # Forward as normal
            
            # We only take out the activations if the activations 
            # are outputs of the chosen layers.
            if layer_num in self.chosen_layers:
                activations.append(x)

        return activations

# vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
# vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# class Normalization(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalization, self).__init__()
#         # .view the mean and std to make them [C x 1 x 1] so that they can
#         # directly work with image Tensor of shape [B x C x H x W].
#         # B is batch size. C is number of channels. H is height and W is width.
#         self.mean = torch.tensor(mean).view(-1, 1, 1)
#         self.std = torch.tensor(std).view(-1, 1, 1)

#     def forward(self, img):
#         # normalize ``img``
#         return (img - self.mean) / self.std

# %%
imsize = (400, 600)
transform = transforms.Compose([
    transforms.Resize(size=imsize),
    transforms.ToTensor()
])

def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)

# %%

# cat : bamboo ratio = 1e6 : 1
# cat : mosaic-1 ratio = 5e4 : 1
# dancing : picasso ratio = 1 : 1e-6
# uog-cloister : romantic ratio = 5e4 : 1
# uog-cloister : mosaic-2 ratio = 5e5 : 1

content_name = 'uog-cloister'
# content_name = 'happy_new_year'
content_path = './' + content_name + '.jpg'
style_name = 'happy_new_year'
# style_name = 'uog-cloister'
style_path = './' + style_name + '.jpg'
content_img = load_image(content_path)
style_img = load_image(style_path)

content_weight = float(1e3)
style_weight = 1

content_img_np = content_img.squeeze().permute((1, 2, 0)).to('cpu').detach().numpy()
style_img_np = style_img.squeeze().permute((1, 2, 0)).to('cpu').detach().numpy()
plt.imshow(np.concatenate((content_img_np, 
                           np.ones(shape=(imsize[0], int(0.1*imsize[1]), 3)), 
                           style_img_np), axis=1))

generated_img = content_img.clone().requires_grad_(True)
# %%
EPOCHS = 300
n_saves = 20
every_save = int(EPOCHS // n_saves)

model = VGG().to(device=device).eval()
model.requires_grad_(False)
optimizer = optim.LBFGS([generated_img])

# %%
img_count = [1]
run = [0]
mse = nn.MSELoss()
print("Optimizing ...")
while run[0] <= EPOCHS:
    def closure():
        with torch.no_grad():
            generated_img.clamp_(0, 1)
        optimizer.zero_grad()

        generated_layer_ft = model(generated_img)
        content_layer_ft = model(content_img)
        style_layer_ft = model(style_img)


        content_loss, style_loss = 0.0, 0.0        
        for j in range(len(generated_layer_ft)):
            generated_features = generated_layer_ft[j]
            content_features = content_layer_ft[j]
            style_features = style_layer_ft[j]

            _, channel, height, width = content_features.shape
            content_loss += mse(generated_features, content_features)

            # Compute the gram matrix for computing style loss
            A = generated_features.view(channel, height * width)
            Gram_generated = torch.matmul(A, A.t())

            A = style_features.view(channel, height * width)
            Gram_style = torch.matmul(A, A.t())

            style_loss += mse(Gram_generated, Gram_style)
            
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()

        run[0] += 1

        if run[0] % every_save == 0:
            print(f"Epoch [{run[0]}/{EPOCHS}]")
            print(f"style loss = {style_loss.item():4f} | content loss = {content_loss.item():4f}")
            save_image(generated_img,
                   f"./generated/{content_name}+{style_name}-{img_count[0]:02d}.png")
            img_count[0] += 1
        return content_weight * content_loss + style_weight * style_loss

    optimizer.step(closure)

    # Last correction
    with torch.no_grad():
        generated_img.clamp_(0, 1)
# %%
