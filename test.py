import torch
import os
from PIL import Image
from models import Decom_Net, Enhance_Net

def normalize_img(img):
    return (img + 1.) / 2.

def denormalize_img(img):
    return (img - 0.5) * 2.

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def process_img(img):
    from torchvision import transforms
    # image transformation
    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, dim=0).cuda()
    return img

def _test(model_name, save_dir, test_data_dir):
    checkpoint = torch.load(model_name)
    Dec.load_state_dict(checkpoint["Dec_model"])
    Enh.load_state_dict(checkpoint["Enh_model"])

    from torchvision import transforms
    for root, _, img_paths in os.walk(test_data_dir):
        for img_path in img_paths:
            with torch.cuda.device(0):
                img = Image.open(os.path.join(root, img_path)).convert('RGB')
                img = process_img(img)
                img = normalize_img(img)
            # decompose
            R, I = Dec(img)
            # enhance
            I_hat = Enh(R, I)
            # enlight
            S_hat = R.mul(I_hat)
            S_hat = denormalize_img(to_numpy(S_hat))
            # transform image array from (-1,1) back to Image
            S_hat = transforms.ToPILImage()(torch.Tensor(S_hat[0]))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            S_hat.save(save_dir + img_path)

if __name__ == "__main__":
    num_layer = 5
    model_name = "./checkpoints/model80.tar" # change checkpoint
    test_data_dir = "../test_dataset/testA"
    save_dir = "../test_dataset/resultsA/"
    # init networks
    Dec = torch.nn.DataParallel(Decom_Net(num_layer).cuda(), device_ids=range(torch.cuda.device_count()))
    Enh = torch.nn.DataParallel(Enhance_Net().cuda(), device_ids=range(torch.cuda.device_count()))
    # test
    _test(model_name, save_dir, test_data_dir)

