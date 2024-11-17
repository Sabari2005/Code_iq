# CODE IQ - ICFAI TECH HACKATHON CHALLENGE
## SOLUTION BY TEAM CODE BLENDERS



## Table of contents

- [Overview](#overview)
- [Architecture Diagram](#my-process)
- [Built with](#built-with)
- [Installation](#installation)
- [Project structure](#structure)
- [Result Analysis](#resultanalysis)
- [Feature](#features)
- [Author](#author)


## Overview
- This is a solution to the [CODE IQ - ICFAI tech Hackathon Challenge (Track 1-2) ](https://unstop.com/hackathons/ml-core-icfai-tripura-1203996). 
- **Track 1: Crowd Counting** - Our model is designed to accurately estimate crowd sizes in images and videos, even under challenging conditions like varying densities, lighting, and diverse environments. We trained a deep learning model that detects and counts individuals efficiently by learning spatial and contextual cues within each frame.



## Architecture Diagram

<img src="./assets/img/architecture.jpg">

## Built with

- ### Frontend:
  - HTML, CSS, JS

- ### Backend:
  - FastAPI
  - Python
  - YOLOv11
  - Deepsort tracking algorithm
- ### Libraries
  - `Ultralytics`
  - `Opencv-python`
  - `numpy`, `pandas` for data handling
  - `uvicorn` for FastAPI

## Installation

### Prerequirements
  - `python3.11`

### Installation steps

  ```
    git clone https://github.com/Sabari2005/Code_iq
    cd Code_iq
  ```
  ```
  pip install -r requirements.txt
  ```


  - Execute each commands in a seperate terminal
  ```
  python index.py
  python serve.py

  ```
  - Open ` http://127.0.0.1:8000` in your browser

## Project structure

```
├──          
├── static
│   ├── css                    
│   └── images                 
├── templates
│   └── index.html             
├── index.py   
├── serve.py                           
├── requirements.txt           
└── README.md                  
```
## Result Analysis

- ### Crowd Counting 
    ![](assets/img/f1.jpg) 
    ![](assets/img/p_curve.jpg)

- ### CustomCSRNet
    <img src="./assets/img/custon CSRNet.jpg">


## Sample model Output
- ### People counting
  ![](assets/img/image42.jpg) 
- ### CSRNet
  <div style="width:100%;height:500px;display:flex;gap:10px">
    <img src="./assets/img/0_Q7dF5t_bAdUW-z1h.webp" style="width:50%;object-fit:contain">
    <img src="./assets/img/download.jpg" style="width:50%;object-fit:contain">
  </div>

## Website Overview
<img src="./assets/img/web1.png">
<div style="width:100%;height:500px;display:flex;gap:10px">
    <img src="./assets/img/web3.png" style="width:50%;object-fit:contain">
    <img src="./assets/img/web4.png" style="width:50%;object-fit:contain">
</div>

## Features
- ### Our website have a real-time corwd prediction streaming

  - Install   ```IP webcam ``` in your android phone and click ```start server ```
  - Now enter the  ``` ipaddress``` and ``` port number ``` in the website 

  <img src="./assets/img/web2.png">


## Author

- Sabari Vadivelan S (Team Leader) - Contact Gmail [sabari132005@gmail.com]()
- Kamal M (Member 1)
- Uvarajan D (Member 2)
- Kaviarasu K (Member 3)