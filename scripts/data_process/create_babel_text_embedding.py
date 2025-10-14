import joblib
import torch
import clip 
import glob
import os
from tqdm import tqdm
import numpy as np

def load_and_freeze_clip(clip_version, device):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

def encode_text(clip_model, raw_text, force_empty_zero=True):
    device = next(clip_model.parameters()).device
    # raw_text - list (batch_size length) of strings with input text prompts
    texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length]
    text_embedding = clip_model.encode_text(texts).float() # [bs, 512]
    if force_empty_zero:  # force empty string to have zero embedding, same as being masked out in original MDM
        empty_text = [text == '' for text in raw_text]
        text_embedding[empty_text, :] = 0
    return text_embedding

device = "cuda:0"

data_dir = "output/HumanoidIm/phc_comp_3/noise_tracking_data/babel/000/"
data_path_list = glob.glob(os.path.join(data_dir, "*.pkl"))

data_path_list = list(set(data_path_list) - set(glob.glob(os.path.join(data_dir, "text*.pkl"))))

clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device=device)

raw_texts_all = []
for path in tqdm(data_path_list):
      data = joblib.load(path)
      if 'frame_labels_all' in data:
        raw_texts_all.extend([seg['proc_label'] for i in data['succ_idxes'] for seg in data['frame_labels_all'][i]])

raw_texts = list(set(raw_texts_all))
num_texts = len(raw_texts)
print('num of unique texts: ', len(raw_texts))
import ipdb; ipdb.set_trace()

text_embeddings = []
batch_start_idx = 0
text_embedding_dict = {}
embedding_path = os.path.join(data_dir, "text_embedding_dict_clip.pkl")
while batch_start_idx < num_texts:
    batch_end_idx = min(batch_start_idx + 256, num_texts)
    text_embeddings.append(encode_text(clip_model, raw_texts[batch_start_idx:batch_end_idx]))
    batch_start_idx = batch_end_idx
text_embeddings = torch.cat(text_embeddings, dim=0).detach().cpu().numpy()
print(text_embeddings.shape)
text_embedding_dict = {raw_texts[idx]: text_embeddings[idx] for idx in range(num_texts)}
text_embedding_dict[''] = np.zeros(512).astype(np.float32)  # for empty text have zero embedding, compatible with mdm text masking
joblib.dump(text_embedding_dict, embedding_path)


