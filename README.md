# sclassifier
This python module allows to perform radio source classification analysis using different ML methods in a supervised/self-supervised or unsupervised way: 
* convolutional neural networks (CNNs)    
* convolutional autoencoders (CAEs)   
* decision trees & LightGBM  
* HDBSCAN clustering algorithm   
* UMAP dimensionality reduction   
* SimCLR & BYOL self-supervised frameworks   

## **Status**
This software is under development. It requires python3 + tensorflow 2.x. 

## **Credit**
This software is distributed with GPLv3 license. If you use it for your research, please add repository link or acknowledge authors in your papers.   

## **Installation**  

To build and install the package:    

* Clone this repository:   
  ```git clone ```
* Create a virtual environment with your preferred python version (e.g. python3.6) in a local install directory (e.g. INSTALL_DIR):   
  ```python3.6 -m venv $INSTALL_DIR```   
* Activate your virtual environment:   
  ```source $INSTALL_DIR/bin/activate```
* Add installation path to your ```PYTHONPATH``` environment variable:   
  ``` export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/lib/python3.6/site-packages ```
* Build and install package:   
  ``` python3.6 setup.py sdist bdist_wheel```    
  ``` python3.6 setup build```   
  ``` python3.6 setup install --prefix=$INSTALL_DIR```   
