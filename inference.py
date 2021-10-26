import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def run_image_classification(model, transform, classes, topk=5):
    """Inference
    """
    testing_dir = 'testing_images\\'
    testing_seq = 'testing_img_order.txt'
    testing_output = 'output/answer.txt'
    fo = open(testing_output, 'w')
    i = 0
    model = model.to(device)
    model.eval()
    with open(testing_seq) as f:
        lines = f.readlines()

        for line in lines:
            i = i + 1
            if (i == 10):
                pass
            # Read image and run prepro
            image = Image.open(testing_dir + line[:-1]).convert("RGB")
            image_tensor = transform(image)
            print(
                f"\n\nImage size after transformation: {image_tensor.size()}")

            image_tensor = image_tensor.unsqueeze(0)
            print(f"Image size after unsqueezing: {image_tensor.size()}")

            # test
            image_tensor = image_tensor.to(device)

            # Feed input
            output = model(image_tensor)
            print(f"Output size: {output.size()}")

            output = output.squeeze()
            print(f"Output size after squeezing: {output.size()}")

            # Result postpro
            _, indices = torch.sort(output, descending=True)
            probs = F.softmax(output, dim=-1)

            fo.write(line[:-1] + ' ' + classes[indices[0]] + '\n')

    fo.close()

    return


def getClasses():
    lines = []
    classes = []

    with open('classes.txt') as f:
        lines = f.readlines()

    for line in lines:
        classes.append(line[:-1])

    return classes


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

classes = getClasses()

model = torch.load('model/myBestModel.pth')

with torch.no_grad():
    run_image_classification(model, inference_transform, classes)
