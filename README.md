

[TOC]



# Object-Detection-pipeline-on-BDD100K-dataset



---

## Input data analysis

---



### Command to build the Docker image

```shell
# Git clone the repository
git clone https://github.com/gaureshshirodkar/Object-Detection-pipeline-on-BDD100K-dataset.git

# Navigate to input analysis directory
cd Object-Detection-pipeline-on-BDD100K-dataset\input_analysis

# To create docker container 
docker build -t input-data-analysis .

# To save docker container image
docker save -o input_analysis.tar input_data_analysis

# To run the docker container
# path_to_data: Path where the training and validation dataset json files are saved
# bdd100k_labels_images_train.json: Training labels json file name
# bdd100k_labels_images_val.json: Validation labels json file name
docker run --rm -p 8050:8050 -v "C:\path_to_data:/data" -e TRAIN_LABELS_PATH=/data/bdd100k_labels_images_train.json -e VAL_LABELS_PATH=/data/bdd100k_labels_images_val.json input_data_analysis
```



### Command to load the exiting image and testing

```sh
# Load the image file
docker load -i input_data_analysis.tar

# To run the docker container
# path_to_data: Path where the training and validation dataset json files are saved
# bdd100k_labels_images_train.json: Training labels json file name
# bdd100k_labels_images_val.json: Validation labels json file name
docker run --rm -p 8050:8050 -v "C:\path_to_data:/data" -e TRAIN_LABELS_PATH=/data/bdd100k_labels_images_train.json -e VAL_LABELS_PATH=/data/bdd100k_labels_images_val.json input_data_analysis
```



### BDD100K input analysis dashboard output

![input_analysis_video](utils/input_analysis_video.gif)





### Conclusion

- Data balance is observed in terms of classes, weather distribution, time of the day and the object frequency.

- Also observed that the data is spread across whole image as shown in Object Position Distribution

- Some of the classes have very high average size. Like bus and train

- To handle these issues following measure can be taken

  - Acquire more images of the classes which are less

  - Data augmentation of the images which has the lesser present classes

  - Handle the data imbalance during model training using the weighted loss function

    





---

## Model Inference

---











---

## Output Analysis

---

