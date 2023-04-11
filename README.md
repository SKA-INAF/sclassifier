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

* Clone this repository in a local directory (e.g. $SRC_DIR):   
  ```git clone https://github.com/SKA-INAF/sclassifier.git```
* Create a virtual environment with your preferred python version (e.g. python3.6) in a local install directory (e.g. INSTALL_DIR):   
  ``` python3.6 -m venv $INSTALL_DIR```   
* Activate your virtual environment:   
  ```source $INSTALL_DIR/bin/activate```
* Install module dependencies listed in ```requirements.txt```:    
  ``` pip install -r requirements.txt```  
* Build and install package:   
  ``` python setup build```   
  ``` python setup install```   
* If required (e.g. outside virtual env), add installation path to your ```PYTHONPATH``` environment variable:   
  ``` export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/lib/python3.6/site-packages ```
