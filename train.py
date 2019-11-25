from models import Decom_Net, Enhance_Net
import torch
import os
nn = torch.nn

# ---------------------------------------Define Network and Init------------------------------
epoch = 200
lr = 1e-4
nums_layer = 5
load_from_check_point = False # set false to train from the scratch; otherwise set iter num to resume training
Dec = Decom_Net(nums_layer).cuda()
Enh = Enhance_Net().cuda()
Dec = torch.nn.DataParallel(Dec, device_ids=range(torch.cuda.device_count()))
Enh = torch.nn.DataParallel(Enh, device_ids=range(torch.cuda.device_count()))

opt_Dec = torch.optim.Adam(Dec.parameters(), lr=lr)
opt_Enh = torch.optim.Adam(Enh.parameters(), lr=lr)

def load_check_point(param):
    if not param:
        return
    else:
        model_name = "./checkpoints/model{}.tar".format(param)
        checkpoint = torch.load(model_name)
        Dec.load_state_dict(checkpoint["Dec_model"])
        Enh.load_state_dict(checkpoint["Enh_model"])

load_check_point(load_from_check_point)

def reconst_loss(x, y):
    return torch.mean(torch.abs(x - y))

def normalize_img(img):
    # from (-1,1) to (0,1)
    return (img + 1.) / 2.

# ----------------------------------------Training---------------------------------------------
def train(epoch):
    from datapipline import Get_paired_dataset
    dataset = Get_paired_dataset(1)
    Dec.train()
    Enh.train()
    flag = True
    for e in range(epoch):
        # train one epoch
        for data in dataset:
            # Get paired data
            with torch.cuda.device(0):
                S_low = data['A'].cuda()
                S_normal = data['B'].cuda()
            S_low = normalize_img(S_low)
            S_normal = normalize_img(S_normal)
            # Decompose Stage
            R_low, I_low = Dec(S_low)
            R_normal, I_normal = Dec(S_normal)

            # Enhance stage
            I_low_hat = Enh(R_low, I_low)
            # ---------------------------------Define Loss Function-------------------------------------
            # Decompose Net Loss: L_reconst + L_invariable_reflectance
            loss_reconst_dec = reconst_loss(S_low, R_low.mul(I_low)) \
                        + reconst_loss(S_normal, R_normal.mul(I_normal)) \
                        + 0.001 * reconst_loss(S_low, R_normal.mul(I_low)) \
                        + 0.001 * reconst_loss(S_normal, R_low.mul(I_normal))
            loss_ivref = 0.01 * reconst_loss(R_low, R_normal)
            loss_dec = loss_reconst_dec + loss_ivref

            def get_smooth(I, direction):
                    #smooth
                    weights = torch.tensor([[0., 0.],
                                            [-1., 1.]]
                                           ).cuda()
                    weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
                    weights_y = torch.transpose(weights_x, 0, 1)
                    if direction == 'x':
                        weights = weights_x
                    elif direction == 'y':
                        weights = weights_y

                    F = torch.nn.functional
                    output = torch.abs(F.conv2d(I, weights, stride=1, padding=1))  # stride, padding
                    return output

            def avg(R, direction):
                return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(R, direction))

            def get_gradients_loss(I, R):
                R_gray = torch.mean(R, dim=1, keepdim=True)
                gradients_I_x = get_smooth(I,'x')
                gradients_I_y = get_smooth(I,'y')

                return torch.mean(gradients_I_x * torch.exp(-10 * avg(R_gray, 'x')) + gradients_I_y * torch.exp(-10 * avg(R_gray, 'y')))

            smooth_loss_low = get_gradients_loss(I_low, R_low)
            smooth_loss_normal = get_gradients_loss(I_normal, R_normal)
            smooth_loss_low_hat = get_gradients_loss(I_low_hat, R_low)

            loss_dec += 0.1 * smooth_loss_low + 0.1 * smooth_loss_normal
            if flag:
                opt_Dec.zero_grad()
                loss_dec.backward()
                opt_Dec.step()

            elif not flag:
                loss_reconst_enh = reconst_loss(S_normal, R_low.mul(I_low_hat))
                loss_enh = loss_reconst_enh + 3 * smooth_loss_low_hat

                opt_Enh.zero_grad()
                loss_enh.backward()
                opt_Enh.step()

            flag = not flag

        print("Epoch: {}; Loss_Dec: {}; Loss_Enh: {}".format(e, loss_dec, loss_enh))
        if e % 10 == 0 and e != 0:
            if not os.path.isdir("./checkpoints/"):
                os.makedirs("./checkpoints/")
            torch.save({"Dec_model": Dec.state_dict(),
                        "Enh_model": Enh.state_dict()},
                        "./checkpoints/model{}.tar".format(e))

            torch.save({"Dec_model": Dec.state_dict(),
                        "Enh_model": Enh.state_dict()},
                       "./checkpoints/model_newest.tar")


if __name__ == '__main__':
    train(epoch)