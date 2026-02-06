# CS50W Project 5 - Traffic Checklist

Complete the implementation of load_data and get_model in traffic.py.

- [x] The load_data function should accept as an argument data_dir, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
    - [x] You may assume that data_dir will contain one directory named after each category, numbered 0 through NUM_CATEGORIES - 1. Inside each category directory will be some number of image files.
    - [x] Use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width IMG_WIDTH and height IMG_HEIGHT.
    - [x] The function should return a tuple (images, labels). images should be a list of all of the images in the data set, where each image is represented as a numpy.ndarray of the appropriate size. labels should be a list of integers, representing the category number for each of the corresponding images in the images list.
    - [x] Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows. Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.

- [x] The get_model function should return a compiled neural network model.
    - [x] You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
    - [x] The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.
    - [x] The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
        - [x] different numbers of convolutional and pooling layers
        - [x] different numbers and sizes of filters for convolutional layers
        - [x] different pool sizes for pooling layers
        - [x] different numbers and sizes of hidden layers
        - [x] dropout

- [x] In a separate file called README.md, document (in at least a paragraph or two) your experimentation process. What did you try? What worked well? What didn’t work well? What did you notice?

Ultimately, much of this project is about exploring documentation and investigating different options in cv2 and tensorflow and seeing what results you get when you try them!

You should not modify anything else in traffic.py other than the functions the specification calls for you to implement, though you may write additional functions and/or import other Python standard library modules. You may also import numpy or pandas, if familiar with them, but you should not use any other third-party Python modules. You may modify the global variables defined at the top of the file to test your program with other values.

[**Dataset link**](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip)

---
# Traffic Sign Classification

## Overview

This project implements a CNN to classify traffic signs from the GTSRB dataset into 43 different categories (0-42)

## Implementation

### Architecture

**Convolutional layer:** 32 filters with 3x3 kernel and ReLU activation
```
model.add(tf.keras.layers.Conv2D(
    32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
))
```

**Max pooling layer:** 2x2 pool size to reduce spatial dimensions
```
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
```

**Flatten Layer:** Converts 2D feature to 1D feature (Vector)
```
model.add(tf.keras.layers.Flatten())
```

**Dense Layer:** 128 units with ReLU activation
```
model.add(tf.keras.layers.Dense(128, activation="relu"))
```

**Dropout Layer:** 0.5 dropout rate to prevent overfitting
```
model.add(tf.keras.layers.Dropout(0.5))
```

**Output Layer:** 43 units with softmax
```
NUM_CATEGORIES = 43

...

model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
```

**Compilation**
- Optimizer: Adam
- Loss function: Categorical crossentropy
- Metrics: Accuracy
```
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Data Processing

Images loaded using cv2 and preprocessed as follows:

- Resized to 30x30 pixels

- Normalized by dividing pixel values by 255 

- Organized by category folders (0-42)

```
IMG_WIDTH = 30
IMG_HEIGHT = 30

...

for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
        
            # Load and resize
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Normalize pixel values
            img = img / 255.0

            # Add to lists
            images.append(img)
            labels.append(category)
    
    return(images, labels)
```

## Experimentation

### Initial Attempt 

The first implementation loaded and resized images only (no normalization), this gave a result of around:

- Training accuracy: 5.5%
- Test accuracy: 5.5%
- Loss remained around 3.5

The model essentially failed, performing really poorly.

### Solution - Adding Normalization

After adding normalization of the images (`img = img / 255.0`), performance improved drastically to:

- Training accuracy: Progressed from around 20% to 90% over 10 epochs.
- Test accuracy: 95%+
- Test loss: 0.125

### Observations

The model demonstrated steady learning across all 10 epochs with accuracy climbing from around 20% to 90% and loss decreasing from 3.00 to 0.33, while test accuracy (95%+) exceeded training accuracy, indicating excellent generalization without overfitting. This project reinforced that proper data preprocessing, particularly normalization, is often more impactful than architectural complexity. With each epoch taking only 4-8 seconds and achieving 97.26% accuracy, the results demonstrate that effective models don't always require extensive computational resources and that this relatively simple CNN could be practically useful for real-world traffic sign recognition tasks.

```
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 7s 10ms/step - accuracy: 0.2330 - loss: 2.9595     
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.6115 - loss: 1.3067 
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 7s 14ms/step - accuracy: 0.7425 - loss: 0.8447 
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 8s 9ms/step - accuracy: 0.7995 - loss: 0.6684  
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.8297 - loss: 0.5444  
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.8512 - loss: 0.4713  
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.8671 - loss: 0.4184  
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.8712 - loss: 0.4082  
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.8849 - loss: 0.3712    
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.8974 - loss: 0.3278  
333/333 - 1s - 4ms/step - accuracy: 0.9726 - loss: 0.1253
``` 

### Conclusion

The final model achieves 97.26% accuracy on the test set with a simple architecture, demonstrating that proper data preprocessing and a well-chosen basic CNN can solve complex image classification problems effectively. 

**Important Lesson learned:** always normalize your image data before feeding it into a neural network...
