#%%
from backbone import Backbone
from transforms import Transforms
import json
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
from pathlib import Path
from time import perf_counter
#%%
def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    return classes
#%%
def load_model(cls, device):
    backbone = Backbone(len(cls))
    model = backbone.mobileNetV3(pretrained=False)
    model_weights = torch.load(f'./result/models/{backbone.get_model_type()}/best_model.pt')
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    return model
#%%
def load_image(path, device):
    pil_img = Image.open(path).convert('RGB')
    transform = Transforms(is_train=False).get_transform()
    img = transform(pil_img).unsqueeze(0)
    img_tensor = img.to(device)
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    rgb_img = np.array(pil_img.resize((w, h))).astype(np.float32) / 255.0
    return img_tensor , rgb_img
#%%
@torch.no_grad()
def predict(model, img, cls):
    output = model(img)
    output = torch.argmax(output, dim=1)
    pred = cls[str(output.item())]
    return pred

def grad_cam(model, img_tensor, rgb_img, target_layers, save_path='inference/gradcam_result.jpg'):
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay


#%%
if __name__ == '__main__':
    img_path = Path(r"Dataset\Pepper__bell___healthy\06f73502-5694-494a-b3cd-d4bf7bf75172___JR_HL 7692.JPG")

    img_name = f"{img_path.stem}_gradcam_result.jpg"

    save_path = Path('inference') / img_name

    classes = load_classes('preProcess/classes.json')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(classes, device)

    img_tensor, rgb_img = load_image(img_path, device)

    prediction = predict(model, img_tensor, classes)

    print(prediction)

    grad_cam(model, img_tensor, rgb_img, [model.blocks[-1][-1]], save_path = save_path)




