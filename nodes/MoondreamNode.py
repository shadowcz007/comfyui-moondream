import os,sys
import folder_paths
from PIL import Image
import numpy as np

import torch
from comfy.model_management import get_torch_device
from huggingface_hub import snapshot_download

# print('#######s',os.path.join(__file__,'../'))

sys.path.append(os.path.join(__file__,'../../'))
                
from moondream import VisionEncoder, TextModel
# from transformers import TextIteratorStreamer

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class MoondreamNode:
    def __init__(self):
        self.torch_device = get_torch_device()
        self.vision_encoder = None
        self.text_model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
             "question": ("STRING", 
                         {
                            "multiline": True, 
                            "default": '',
                            "dynamicPrompts": False
                          }),
                             },

            
                }
    
    RETURN_TYPES = ("STRING",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Prompt"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
  
    def run(self,image,question):
      
        result=[]
        if self.vision_encoder==None:
            model_path=os.path.join(__file__,'../../checkpoints')
            if os.path.exists(model_path)==False:
                model_path = snapshot_download("vikhyatk/moondream1",local_dir='./checkpoints',endpoint='https://hf-mirror.com')
            self.vision_encoder = VisionEncoder(model_path)
            self.text_model = TextModel(model_path)
         
        question=question[0]

        for i in range(len(image)):
            im=image[i]
            im=tensor2pil(im)
             
            image_embeds = self.vision_encoder(im)
            res=self.text_model.answer_question(image_embeds, question)
         
            result.append(res)
        
        return (result,)