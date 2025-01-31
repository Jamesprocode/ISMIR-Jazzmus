<p align="center">
  <a href=""><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
  <a href=""><img src="https://imgur.com/H2YEzDY.png" alt="JKU-logo" width="40"></a>
</p>

<h1 align="center">Jazzmus</h1>

<h3 align="center">Optical Music Recognition for jazz lead sheets</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-orange.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" alt="Lightning">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch 2.5.1">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a>
</p>

## About

abc

## How To Use

### Prepare data
Install dependencies with 
```python
pip install -r requirements.txt
python3 utils/prepare_hf_dataset.py
```

This script will generate a folder structure in data/ with the name of the dataset (jazzmus) and its regions, annotations and splits.

### Train
```python
bash train.sh
```

## Cite

## License
This work is under a [MIT](LICENSE) license.


