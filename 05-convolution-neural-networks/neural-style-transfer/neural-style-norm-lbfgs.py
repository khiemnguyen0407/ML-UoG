# %%
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

# %%
device = 'cuda'

# image = Image.open('./uog-cloister.jpg')

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

content_name = 'uog-cloister'
content_path = './' + content_name + '.jpg'
style_name = 'romantic'
style_path = './' + style_name + '.jpg'
content_img = load_image(content_path)
style_img = load_image(style_path)

content_img_np = content_img.squeeze().permute((1, 2, 0)).to('cpu').detach().numpy()
style_img_np = style_img.squeeze().permute((1, 2, 0)).to('cpu').detach().numpy()
plt.imshow(np.concatenate((content_img_np, style_img_np), axis=1))

# %%

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std
    
class NeuralStyle(nn.Module):
    def __init__(self, norm_mean, norm_std):

        super(NeuralStyle, self).__init__()
        self.normalization = Normalization(norm_mean, norm_std)

        self.features = models.vgg19(weights='DEFAULT').features[:29]

        self.chosen_layers = [0, 5, 10, 19, 28]

    def forward(self, x):
        features = []
        # x = self.normalization(x)
        for layer_num, layer in enumerate(self.features):
            x = layer(x)
            
            if layer_num in self.chosen_layers:
                features.append(x)

        return features
# %%
norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
norm_std = torch.tensor([0.229, 0.224, 0.225], device=device)
model = NeuralStyle(norm_mean, norm_std).to(device=device).eval()
model.requires_grad_(False)

generated_img = content_img.clone().requires_grad_(True)
optimizer = optim.LBFGS([generated_img], lr=1.5)
with torch.no_grad():
    generated_img.clamp_(0, 1)

# output = model(generated_img)
# print(f"len(output) = {len(output)}")

EPOCHS = 200
content_weight = float(1)
style_weight = float(5e4)
n_saves = 10
every_save = int(EPOCHS // n_saves)
mse = nn.MSELoss()
# cat : bamboo ratio = 1e6 : 1
# cat : mosaic ratio = 5e4 : 1
# dancing : picasso ratio = 1 : 1e-6
# %%
img_count = [1]
run = [0]
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
            HW = height * width
            A = generated_features.view(channel, HW)
            Gram_generated = torch.matmul(A, A.t())
            # Gram_generated = Gram_generated.div(channel * HW)

            A = style_features.view(channel, HW)
            Gram_style = torch.matmul(A, A.t())
            # Gram_style = Gram_style.div(channel * HW)

            style_loss += mse(Gram_generated, Gram_style)
            
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()

        run[0] += 1

        if run[0] % every_save == 0:
            print(f"Epoch [{run[0]}/{EPOCHS}]")
            print(f"style loss = {style_loss.item():.4f} | content loss = {content_loss.item():.4f}")
            save_image(generated_img,
                   f"./generated/{content_name}+{style_name}-{img_count[0]:02d}.png")
            img_count[0] += 1
        return content_weight * content_loss + style_weight * style_loss

    optimizer.step(closure)

    # Last correction
    with torch.no_grad():
        generated_img.clamp_(0, 1)
# %%
