from PIL import Image
import os
import numpy as np
import torch
from deform2self import deform2self, deform2self_sequence
from utils import add_gaussian_noise, img_float2int
import yaml


if __name__ == '__main__':

    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


    # check if gpu with index 0 is available
    if not torch.cuda.is_available():
        print("GPU not available, exiting...")
        exit()

    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu']

    print('opened YAML')

    imgs = []
    for fname in sorted(os.listdir('./data')):
        imgs.append(np.array(Image.open(os.path.join('./data', fname))))
    imgs = np.stack(imgs, axis=0).astype(np.float32)
    imgs /= imgs.max()
    imgs = torch.from_numpy(imgs).unsqueeze(1) # Tx1xHxW

    print('loaded images')

    noisy = add_gaussian_noise(imgs, 15)

    print('added noise')
    
    denoised = deform2self_sequence(noisy, config) # Tx1xHxW

    print('denoised images')

    denoised = denoised.squeeze(1).cpu().numpy()
    denoised = img_float2int(denoised)

    noisy = noisy.squeeze(1).numpy()
    noisy = img_float2int(noisy)

    print('saving images')

    for i, (img_n, img_d) in enumerate(zip(noisy, denoised)):
        Image.fromarray(img_n).save('outputs/noisy%d.png' % i)
        Image.fromarray(img_d).save('outputs/denoised%d.png' % i)
    
    print('saved images')


# ---------------------------- ERROR ----------------------------
"""
Traceback (most recent call last):
  File "/home/zimadmin/Documents/deconvolution/deform2self/Deform2Self/demo.py", line 42, in <module>
    denoised = deform2self_sequence(noisy, config) # Tx1xHxW
  File "/home/zimadmin/Documents/deconvolution/deform2self/Deform2Self/deform2self.py", line 75, in deform2self_sequence
    denoised.append(deform2self(noisy[i_min:i_max], i-i_min, config))
  File "/home/zimadmin/Documents/deconvolution/deform2self/Deform2Self/deform2self.py", line 45, in deform2self
    loss = total_loss(model(noisy_img, idx_tgt), noisy_img, idx_tgt)
  File "/home/zimadmin/anaconda3/envs/deconvolution/lib/python3.8/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/zimadmin/Documents/deconvolution/deform2self/Deform2Self/models.py", line 227, in forward
    denoise1_img = self.denoise1(input_img) # TN x 1 x H x W
  File "/home/zimadmin/anaconda3/envs/deconvolution/lib/python3.8/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/zimadmin/Documents/deconvolution/deform2self/Deform2Self/models.py", line 180, in forward
    x = self.convs[0](x)
  File "/home/zimadmin/anaconda3/envs/deconvolution/lib/python3.8/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/zimadmin/Documents/deconvolution/deform2self/Deform2Self/models.py", line 27, in forward
    x = self.conv(x)
  File "/home/zimadmin/anaconda3/envs/deconvolution/lib/python3.8/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/zimadmin/anaconda3/envs/deconvolution/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 345, in forward
    return self.conv2d_forward(input, self.weight)
  File "/home/zimadmin/anaconda3/envs/deconvolution/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 341, in conv2d_forward
    return F.conv2d(input, weight, self.bias, self.stride,
RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
"""



     