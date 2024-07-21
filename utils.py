import json
import torch
import torchtext
from transformers import AutoModel, AutoTokenizer, AutoProcessor, CLIPModel

def get_torchtext_vectors(torchtext_vocab: torchtext.vocab, words: list) -> torch.tensor:
    """
    Retrieve word embeddings from a torchtext embedding object.
    """
    
    return torch.stack([torchtext_vocab.vectors[torchtext_vocab.stoi[w]] for w in words])

def get_lm_embeddings(model: AutoModel, tokenizer: AutoTokenizer, texts: list) -> torch.tensor:
    """
    Returns the last token language model embeddings for the given texts.
    """

    # Create list to store embeddings
    emb_list = []

    # Iterate through texts and append embeddings to list
    for text in texts:

        # Encode text using tokenizer
        input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')

        # Obtain embeddings for last token
        with torch.no_grad():
            embeddings = model(input_ids)[0]

        last_token_embedding = embeddings[-1][-1]

        # Append to list
        emb_list.append(last_token_embedding)

    return torch.vstack(emb_list)

def get_clip_text_embs(model: CLIPModel, tokenizer: AutoTokenizer, texts: list) -> torch.tensor:
    """
    Returns the text embeddings for the given texts.
    """

    # Tokenize text inputs and obtain text embeddings
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        text_features = model.get_text_features(**inputs)

    return text_features

def get_clip_image_embs(model: CLIPModel, processor: AutoProcessor, images: list) -> torch.tensor:
    """
    Returns the image embeddings for the given images.
    """

    # Preprocess images and obtain image embeddings
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        image_features = model.get_image_features(**inputs)

    return image_features

def read_json_file(file_path: str) -> dict:
    """
    Read a JSON file and return a dictionary.
    """

    with open(file_path, 'r') as f:
        return json.load(f)