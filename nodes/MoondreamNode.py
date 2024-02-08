import os,sys
import folder_paths
from PIL import Image
import numpy as np

import torch
from comfy.model_management import get_torch_device
from huggingface_hub import hf_hub_download

# print('#######s',os.path.join(__file__,'../'))

sys.path.append(os.path.join(__file__,'../../'))
                
# from moondream import VisionEncoder, TextModel,detect_device
# from transformers import TextIteratorStreamer

from moondream import Moondream, detect_device
from threading import Thread
from transformers import TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer



# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class MoondreamNode:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        # get_torch_device()
        self.moondream = None
        self.tokenizer = None
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
            "device": (["cpu","cuda"],),
                             },

            
                }
    
    RETURN_TYPES = ("STRING",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Prompt"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
  
    def run(self,image,question,device):
      
        result=[]

        if device[0]!=self.device.type and device[0]=='cpu':
            self.device=torch.device(device[0])
            self.dtype = torch.float32
            self.moondream =None
        elif device[0]!=self.device.type:
            device, dtype = detect_device()
            self.device=device
            self.dtype =dtype
            self.moondream =None

        if self.moondream ==None:
            model_path=os.path.join(__file__,'../../checkpoints')
            if os.path.exists(model_path)==False:
                model_path = hf_hub_download("vikhyatk/moondream1",
                                               local_dir=model_path,
                                               filename="config.json",
                                               endpoint='https://hf-mirror.com')
                model_path = hf_hub_download("vikhyatk/moondream1",
                                               local_dir=model_path,
                                               filename="model.safetensors",
                                               endpoint='https://hf-mirror.com')
                model_path = hf_hub_download("vikhyatk/moondream1",
                                               local_dir=model_path,
                                               filename="tokenizer.json",
                                               endpoint='https://hf-mirror.com')
            
            self.tokenizer = Tokenizer.from_pretrained(model_path)
            self.moondream = Moondream.from_pretrained(model_path).to(device=self.device, dtype=self.dtype)
            self.moondream.eval()

         
        question=question[0]

        for i in range(len(image)):
            im=image[i]
            im=tensor2pil(im)

            image_embeds = self.moondream.encode_image(im)
            
            # streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
            res=self.moondream.answer_question(image_embeds, question,self.tokenizer)

            result.append(res)

        return (result,)