# Simple Style Transfer in [PyTorch](https://github.com/pytorch/pytorch) 
Add styles from famous paintings to any photo. Checkout the results! <br />
<img src = 'https://user-images.githubusercontent.com/30996098/188305844-ed0618f1-b554-4f5f-bca0-a08ea248c25e.jpg' width = 250 height = 250>
<img src = 'https://user-images.githubusercontent.com/30996098/188305854-c9378897-9b8d-4d6a-91e5-3541306cee0d.jpg' width = 250 height = 250>
<img src = 'https://user-images.githubusercontent.com/30996098/188305862-e14fcad0-1002-40e2-9ce5-1c83e8e5d019.png' width = 250 height = 250>
<img src = 'https://user-images.githubusercontent.com/30996098/188305901-b32fe6a5-630a-480a-9e0f-9adfce2c8396.jpg' width = 250 height = 250>
<img src = 'https://user-images.githubusercontent.com/30996098/188305906-1721e529-a25b-4a45-a4d3-7a37fcebe98f.jpg' width = 250 height = 250>
<img src = 'https://user-images.githubusercontent.com/30996098/188305915-f02ffe40-2a26-4368-8189-8861e81edbd0.png' width = 250 height = 250>

## Introduction
My work implements the paper by Gaytes' [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

## Documentation
Use `nst.py` to train a new style transfer network. Run `python nst.py` to view all the possible parameters. Example usage:
```
python nst.py --content-path path/to/content/img.jpg \
--style-path path/to/style/img.jpg \
--gen-path path/to/save/gen/img.jpg
```
## Requirements
- Torch 1.6.0+cu101 (A decent GPU is needed if you don't wanna train for days)
- Python 3.8+
