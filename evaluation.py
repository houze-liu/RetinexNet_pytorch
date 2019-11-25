from PIL import Image
import torch
import numpy as np
import os
import re


class Evaluation():
    """
    evaluation on two directories with paired images; the paired index should be part of image names
    """
    def __init__(self, root_1, root_2):
        self.path = {}
        self._traverse_test_dir(root_1, self.path)
        self._traverse_test_dir(root_2, self.path)

    def calcualte_psn_socre(self):
        score_list = []
        for item in self.path:
            img_1, img_2 = self.path[item]
            img_1 = Image.open(img_1)
            img_1 = np.array(img_1)
            img_2 = Image.open(img_2)
            img_2 = np.array(img_2)
            img_1 = torch.Tensor(img_1)
            img_2 = torch.Tensor(img_2)

            r = torch.nn.functional.mse_loss(img_1[:, :, 0], img_2[:, :, 0])

            g = torch.nn.functional.mse_loss(img_1[:, :, 1], img_2[:, :, 1])

            b = torch.nn.functional.mse_loss(img_1[:, :, 2], img_2[:, :, 2])

            t = (r + g + b) / 3.
            t = t.data.numpy()
            score_list.append(10. * np.log10(255. ** 2 / t))

        return score_list

    def _traverse_test_dir(self, root_dir, path):
        for (r, v, file_names) in os.walk(root_dir):
            for f in file_names:
                if f.endswith('.png') and not f.startswith("._"):
                    # if not in key(number),then create new list for that key and append path as value
                    # if already exists the key, then append path as value
                    idx = int(re.findall(string=f, pattern='\d+')[0])
                    img_path = os.path.join(r, f)
                    if idx not in path.keys():
                        path[idx] = [img_path]
                    else:
                        path[idx].append(img_path)

root_1 = "../test_dataset/testA/"
root_2 = "../test_dataset/resultsA/"
eval = Evaluation(root_1, root_2)
scores = eval.calcualte_psn_socre()
print(np.mean(scores))