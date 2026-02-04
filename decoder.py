import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
from models.HidingRes import HidingRes
from PIL import Image

def decode_pic(Rnet, test_pic, output_pic):
    model_dir = Rnet
    Rnet = HidingRes(in_c=3, out_c=3)
    Rnet = Rnet.cuda()
    Rnet.load_state_dict(torch.load(model_dir))
    Tensor = torch.cuda.FloatTensor
    loader = transforms.Compose([
        transforms.ToTensor(), ])

    img = Image.open(test_pic)
    img = loader(img)
    img = img.cuda()

    imgv = Variable(img)
    R_img = Rnet(imgv)

    output = R_img.view(1, 3, 512, 512)
    vutils.save_image(output, output_pic, nrow=1, padding=1, normalize=False)

if __name__ == '__main__':
    Rnet = ''
    test_pic = ""
    output_pic = ''
    decode_pic(Rnet, test_pic, output_pic)

