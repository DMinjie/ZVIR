import torch
import os
import color
from PIL import Image
import torchvision.transforms as transforms
from torch.nn.functional import mse_loss
import ZVIR_NET
from pytorch_msssim import SSIM
import torch.nn.functional as F



weight = 0.5
vis_img_path = 'testdata/vi/00810.png'  # vis_path
ir_img_path = 'testdata/ir/00810.png'  # ir_path
output_folder = 'output'









device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use device: {device}")
IR_image=Image.open(ir_img_path).convert('L')
VI_image=Image.open(vis_img_path).convert('L')
image_supervise1 = VI_image
image_supervise2 = IR_image
image_input_A = VI_image
image_input_B = Image.blend(VI_image,IR_image,weight )
image_input_C = Image.open(vis_img_path).convert('RGB')

resize_transform= transforms.Resize((480,640))
image_supervise1 = resize_transform(image_supervise1)
image_supervise2= resize_transform(image_supervise2)
image_input_A = resize_transform(image_input_A)
image_input_B = resize_transform(image_input_B)
image_input_C = resize_transform(image_input_C)

transform = transforms.ToTensor()
supervision_image1 = transform(image_supervise1).unsqueeze(0).to(device)
supervision_image2 = transform(image_supervise2).unsqueeze(0).to(device)
putin_A = transform(image_input_A).unsqueeze(0).to(device)
putin_B= transform(image_input_B).unsqueeze(0).to(device)



def pad(image, padding=(32, 32, 16, 16)):

    return F.pad(image, padding, mode='reflect')


def crop(image, original_height, original_width, padding=(32, 32, 16, 16)):

    _, height, width = image.size()
    top = padding[2]
    left = padding[0]
    bottom = height - padding[3]
    right = width - padding[1]

    return image[:, top:bottom, left:right]

supervision_image1=pad(supervision_image1)
supervision_image2=pad(supervision_image2)
putin_A=pad(putin_A)
putin_B=pad(putin_B)



def sobel(image):
    # Sobel filters for x and y gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        image.device)
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient_magnitude
import time

def train(input_A,input_B,supervise1,supervise2,  num):
    model = ZVIR_NET.ZVIR_NET().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    loss_ssim = SSIM(data_range=1.0, size_average=True, channel=1)
    sobel_1 =sobel(supervise1)
    sobel_2=sobel(supervise2)
    sobel_all=sobel_1+sobel_2

    for i in range(num):
        optimizer.zero_grad()
        output = model(input_A,input_B)

        sobel_out=sobel(output)
        loss = 1 - loss_ssim(output, supervise1)+3*mse_loss(sobel_out,sobel_all)
        loss.backward()
        optimizer.step()
        print(f'Iteration {i},loss{loss}')

    output1 = model(input_B,input_A)
    final_output = output1.detach()
    return final_output



times =200
start_time = time.time()
output = train( putin_A, putin_B,supervision_image1,supervision_image2,  times).to(device)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total Time: {elapsed_time:.2f} second。")

os.makedirs(output_folder, exist_ok=True)


to_pil = transforms.ToPILImage()
resize_transform1 = transforms.Resize((480,640))


for i, image_tensor in enumerate(output):
    image_tensor=crop(image_tensor,480,640)
    image_tensor = image_tensor.clamp(0, 1)
    image_pil = to_pil(image_tensor)
    image_pil = resize_transform1(image_pil)
    image_pil = color.gray_to_rgb_from_reference(image_pil,image_input_C)
    output_path = os.path.join(output_folder, f'blend_image.png')
    image_pil.save(output_path)

torch.cuda.empty_cache()
print(f"已释放本轮显存。")
