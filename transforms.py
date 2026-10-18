from torchvision.transforms import transforms
from backbone import Backbone
import timm

class Transforms:
    def __init__(self, is_train=True):
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.5, 0.5, 0.5] , std = [0.5, 0.5, 0.5]),

            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    def get_transform(self):
        return self.transform




if __name__ == "__main__":
    backbone = Backbone(3)
    model = backbone.levit384(pretrained=True)
    model_config = timm.data.resolve_data_config({}, model=model)
    print(model_config)