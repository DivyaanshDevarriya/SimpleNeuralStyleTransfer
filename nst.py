import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from tqdm import tqdm
from argparse import ArgumentParser
import os

NUM_EPOCHS = 400
LEARNING_RATE = 0.01
ALPHA = 1   # content weight
BETA = 0.005 # style weight

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
image_size = 356

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--content-path', type=str,
                        dest='content_path', help='path to content image',
                        metavar='content-img-path', required=True)

    parser.add_argument('--style-path', type=str,
                        dest='style_path', help='path to style image',
                        metavar='style-img-path', required=True)

    parser.add_argument('--gen-path', type=str,
                        dest='gen_path', help='path to save generated image',
                        metavar='generated-img-path', required=True)

    return parser


def check_opts(opts):
    assert os.path.exists(opts.content_path), "content image path not found!"
    assert os.path.exists(opts.style_path), "style image path not found!"
    assert opts.gen_path != "", "generated image save path not found!"


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


class VGG(nn.Module):
    
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model  = models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


if __name__ == "__main__":

    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    loader = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
    )

    content_img = load_image(options.content_path)
    style_img = load_image(options.style_path)
    generated_img_save_path = options.gen_path

    model = VGG().to(device).eval()

    generated = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr = LEARNING_RATE)

    for step in tqdm(range(NUM_EPOCHS)):
        
        generated_features = model(generated)
        content_features = model(content_img)
        style_features = model(style_img)
        
        style_loss = content_loss = 0
        for gen_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
            
            batch_size, channel, height, width = gen_feature.shape
            
            #caclulate content loss
            content_loss+=torch.mean((gen_feature - content_feature)**2)
            
            #compute gram matrix
            G = gen_feature.view(channel, height*width).mm(
                gen_feature.view(channel, height*width).t())
            
            A = style_feature.view(channel, height*width).mm(
                style_feature.view(channel, height*width).t())
            
            #calculate style loss
            style_loss+=torch.mean((G - A)**2)
        
        total_loss = (ALPHA*content_loss) + (BETA*style_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step%200 == 0:
            print(f'Loss: {total_loss.item()}')
            save_image(generated, generated_img_save_path)

    save_image(generated, generated_img_save_path)    
    print(f'Generated image saved at : {generated_img_save_path}')
    
    torch.cuda.empty_cache()