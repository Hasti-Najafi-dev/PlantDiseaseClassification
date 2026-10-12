from torchvision.transforms import transforms

class Transforms:
    def __init__(self, is_train=True):
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.5 , 0.5 , 0.5] , std = [0.5, 0.5 , 0.5]),

            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    def get_transform(self):
        return self.transform