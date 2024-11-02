# FLIPKART GRID 6.0 - ROBOTICS CHALLENGE
## SOLUTION BY TEAM SABARI132005



## Table of contents

- [Overview](#overview)
- [Screenshot](#screenshot)
- [Architecture Diagram](#my-process)
- [Built with](#built-with)
- [Installation](#installation)
- [Project structure](#structure)
- [Result Analysis](#resultanalysis)
- [Database storage](#database)
- [Author](#author)


## Overview
- This is a solution to the [Smart Vision Technology Quality Control in Flipkart Grid 6.0 - Robotics Challenge](https://unstop.com/hackathons/flipkart-grid-60-robotics-challenge-flipkart-grid-60-flipkart-1024253). 
- Our project is a real-time product detection and freshness monitoring system, leveraging YOLOv8 object detection to identify products, categorize them by brand, and evaluate the freshness of fruits and store it in the database.


## Screenshot
- ### Product detection, recognition and extraction
  <img  src="./assets/image1.jpg">
  <img src="./assets/image2.jpg">

- ### Brand based counting
  <img src="./assets/count1.jpg">
  <img src="./assets/count2.jpg">

- ### Fruit freshness identification
  <img src="./assets/output.jpg">
## Architecture Diagram

![](assets/architecture.jpg)

## Built with

- ### Frontend:
  - HTML, CSS, JS

- ### Backend:
  - FastAPI
  - Python
  - YOLOv8 for object detection
  - YOLOv11 (instant segmentation) for detecting fruit freshness
  - KMeans clustering algorithm
- ### Database
  - Sqlite3
- ### Libraries
  - `Ultralytics`
  - `Opencv-python`
  - `numpy`, `pandas` for data handling
  - `uvicorn` for FastAPI

## Installation

### Prerequrements
  - `python3.11`
  - `DB Browser` - to see the database

### Installation steps

  ```
    git clone
    cd
  ```
  ```
  pip install -r requirements.txt
  ```
  - In the `mode1_product.py`  `line 62` replace your webcam url

![](assets/url1.png)

  - In the `model2_fruit.py` `line 31` replace your webcam url

 ![](assets/url2.png) 


  - Execute each commands in a seperate terminal
  ```
  python index.py
  python model1_fastapi.py
  python model2_fastapi.py
  ```
  - Open ` http://127.0.0.1:8000` in your browser

## Project structure

```
├──          
├── static
│   ├── css                    # Frontend styling files
│   └── images                 # Static images (if any)
├── templates
│   └── index.html             # Main HTML page
├── index.py   
├── model1_fastapi.py          
├── model2_fastapi.py                 
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```
## Result Analysis

- ### Product details recognition and extraction model
    ![](assets/detection.png) 
    ![](assets/product_model.png) 
    ![](assets/results.png) 
- ### Fruit freshness detection model


## Sample model Output
- ### Product details detection and recognition model(real time images)
  ![](assets/real-time.png) 

  ![](assets/real-time2.png) 

## Database Storage

- you can see the product details in 
## Author

- Sabari Vadivelan S (Team Leader) - Contact Gmail [sabari132005@gmail.com]()
- Kamal M (Member 1)
- Uvarajan D (Member 2)
- Kaviarasu K (Member 3)
- Sarathi S (Member 4)