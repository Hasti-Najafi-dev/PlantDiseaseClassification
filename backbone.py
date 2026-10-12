#%%
import timm
from torch.nn import Linear
from transformers import LevitForImageClassification


class Backbone:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model_type = None

    def mobileNetV3(self, pretrained=True):
        model = timm.create_model('mobilenetv3_small_050', pretrained=pretrained)
        model.classifier = Linear(in_features=model.classifier.in_features, out_features=self.num_classes)
        self.model_type = 'mobileNetV3'
        return model
    def levit384(self, pretrained=True):
        model = timm.create_model('levit_384', pretrained=pretrained, num_classes=self.num_classes)
        self.model_type = 'levit384'
        return model

    def get_model_type(self):
        return self.model_type
#%%
# backbone = Backbone(10)
# model = backbone.levit384()
# print(model)