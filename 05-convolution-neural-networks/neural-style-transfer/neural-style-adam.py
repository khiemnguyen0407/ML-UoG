# %%
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# %%
vgg19 = models.vgg19(weights='DEFAULT')
model = vgg19.features
device = 'cuda'

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_layers = [0, 5, 10, 19, 28]


        self.model = models.vgg19(weights='DEFAULT').features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if layer_num in self.chosen_layers:
                features.append(x)

        return features
    
# %%
im_size = (256, 256)
transform = transforms.Compose([
    transforms.Resize(size=im_size),
    transforms.ToTensor()
])

def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)

# %%
content_name = 'dancing'
content_path = './' + content_name + '.jpg'
style_name = 'picasso'
style_path = './' + style_name + '.jpg'
content_img = load_image(content_path)
style_img = load_image(style_path)

generated_img = content_img.clone().requires_grad_(True)
# %%
EPOCHS = 5000
lr = 0.01
alpha = 1.0
beta = 0.05

optimizer = optim.Adam([generated_img], lr=lr)
model = VGG().to(device=device).eval()
# %%
img_count = 1
n_saves = 20
every_save = EPOCHS // n_saves
for epoch in range(EPOCHS):
    generated_layer_ft = model(generated_img)
    content_layer_ft = model(content_img)
    style_layer_ft = model(style_img)

    content_loss, style_loss = 0.0, 0.0

    optimizer.zero_grad()
    for j in range(len(generated_layer_ft)):
        generated_features = generated_layer_ft[j]
        content_features = content_layer_ft[j]
        style_features = style_layer_ft[j]

        batch_size, channel, height, width = content_features.shape
        content_loss += torch.mean((generated_features - content_features)**2)

        # Compute the gram matrix for computing style loss
        A = generated_features.view(channel, height * width)
        Gram_generated = torch.matmul(A, A.t())

        A = style_features.view(channel, height * width)
        Gram_style = torch.matmul(A, A.t())

        style_loss += torch.mean((Gram_generated - Gram_style)**2) 
        
    total_loss = alpha * content_loss + beta * style_loss
    
    
    total_loss.backward()
    optimizer.step()

    if epoch % (every_save) == 0:
        print(f"Epoch [{epoch} / {EPOCHS}]:\n" + 50*"=")
        print(f"content loss: {content_loss.item():6.5f} | style loss: {style_loss.item():6.5f}")
        print(f"total loss: {total_loss.item():6.5f}")
        save_image(generated_img, f"./generated/{content_name}+{style_name}-{img_count:02d}.png")
        img_count += 1
# %%
