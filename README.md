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

<style>
  .spaced-img {
    margin: 0 25px; /* 50px total spacing between images */
  }
</style>

## About

abc

## How To Use

abc


### Predict a sample
Once trained a model, the script will generate the weights and the vocabulary associated to that training, to predict an image, use the weights, the vocabulary and a file where each line represents a path to a staff-level image to predict.
```python
python3 predict.py
```
It accepts multiple params such as: --checkpoint_path weigths/crnn/austrian_0.ckpt --vocab data/vocabs/austrian_ctc_w2i.json --samples <file_path>
This script will generate a predictions.json file with all the images and its corresponding prediction made by the model.

## Cite

## License
This work is under a [MIT](LICENSE) license.


