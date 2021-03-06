# Inferencing a Pre-trained VGG-16 Model for Custom Images

## Main Technical tools
Tensorflow: 1.0.0  
Python: 3.4  

## Other Dependencies
Flask (web service)  
urllib (HTTP request handling)  
PIL (image processing)  

## Available files
imagenet_classes.py - Class names of the imagenet dataset  
vgg_inference.py - The script that loads the model and infer from it and return the class and confidence for an image  
query_service.py - The web service implemented with Flask  
config.py - The constants used by various scripts  
app_unit_tests.py - Unit Testing  
input_url.json - Example JSON file for providing input URLs to the program  

## Introduction
This repository implements a script for inferencing the classes of custom images using a pretrained deep network (VGG-16). VGG-16 is one of the state-of-the art deep models for image classification which achieved ~92% top-5 accuracy on Imagenet classification task. The weights (parameters) are found [here](https://www.cs.toronto.edu/~frossard/post/vgg16/) (vgg16_weights.npz)
The architecture of the model can be found in this [paper](https://arxiv.org/pdf/1409.1556.pdf)  

## How to run
1. Run the script query_service.py 
2. Go to http://127.0.0.1:5000 (For this prototype this will have to run locally)
3. The home page should have a description about what you can do
4. As an example try entering the following in the url bar http://127.0.0.1:5000/infer_from_url?url=https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1.jpg
5. This should return a result similar to 
```
{
  "download_status": {
    "error": "None", 
    "saved_as": "image-0.jpg", 
    "success_status": true, 
    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1.jpg"
  }, 
  "result": {
    "class": "tabby, tabby cat (Save filename: image-0.jpg)", 
    "confidence": 0.419, 
    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1.jpg"
  }
}
```
5. The script might download the weights (~ 500MB) and will require few minutes if not available locally. Once weights are downloaded tensorflow will load those parameters in to tensorflow variables


## Available Options
* Infer with URL
    * ```ip_address/infer_from_url?url=www.example.com/image.jpg```
* Infer with JSON file
   * ```ip_address/infer?filename=example.json```
   * example.json need to reside in your project home folder (where query_service.py is at)
* Infer with JSON file with a confidence threshold
   * ```ip_address/infer_with_conf?filename=example.json&confidence_threshold=0.01```

## Available Functionality
* Can process images in batches when multiple URLs are provided in the input JSON file
* Will use the GPU if available or will fall back to the CPU if a GPU is not detected
* Can use a confidence threshold to discard uncertain inputs
* Loads a deep model with a given weight file (.npz)
* Web service to access the inference capabilities with the loaded model
* Preprocess the given image by resizing the image to a fixed size
* All images needs to be in jpg format but will automatically convert images to jpg for known formats. Otherwise will return an error
* Unit tests are available for the application layer
* Can infer the model with a confidence threshold to not show images with low confidence

## Future Work
* Extend the model to have an effective input reading pipeline, seamlessly integrating data reading in tensorflow
* Extend to read a zipped collection of images
* Divide the data to multiple batches if the number of inputs exceed a threshold (e.g >128 inputs)


