---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 31683955.0
    num_examples: 18
  download_size: 31686909
  dataset_size: 31683955.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
