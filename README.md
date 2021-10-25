# deep learning homework1

# Requirements
python  3.8.11

pytorch 1.9.1

PIL

tqdm

timm

# train.py

train and valid with

# utlis.utlity.py

Use to print hist.

# factory.py

To create pretrained model from timm.

# birdDataset.py

To create bird dataset literally.

# inference.py

to reproduce result, please make sure that below exist:
1. classes.txt
- contain 200 classes
2. testing_img_order.txt
- the testing order, can be found from https://competitions.codalab.org/competitions/35668#participate-get_starting_kit

3. testing_images/
- this folder contain all the testing images.
- You can find images from https://competitions.codalab.org/competitions/35668#participate-get_starting_kit
- And please make sure that folder name is 'testing_images'

4. model/myBestModel.pth
- make sure that folder name is 'model', and model name is 'myBestModel.pth'
- downloadlink:

5. output/
- This folder contain the output. please make sure this folder exist.

with above, you can simply run inference.py to generate 'output/answer.txt'

# Pretrained model

Use Swin-Transformer by Microsoft
https://github.com/microsoft/Swin-Transformer