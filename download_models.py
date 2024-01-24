from huggingface_hub import snapshot_download
import os
local_dir=os.path.join(__file__,'../checkpoints')
print(os.path.exists(local_dir))
model_path = snapshot_download("vikhyatk/moondream1",
                               local_dir=local_dir,
                               endpoint='https://hf-mirror.com')
print(model_path)