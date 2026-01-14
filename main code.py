import cv2
import pickle
import glob
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,SimpleRNN
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from tensorflow.keras.models import load_model

# -------------------------------------------
# Step 1: Load Images
# -------------------------------------------

image_dataset = [
    "Vegetable Images/train/Bean/*.jpg",
    "Vegetable Images/train/Bitter_Gourd/*.jpg",
    "Vegetable Images/train/Bottle_Gourd/*.jpg",
    "Vegetable Images/train/Brinjal/*.jpg",
    "Vegetable Images/train/Broccoli/*.jpg",
    "Vegetable Images/train/Cabbage/*.jpg",
    "Vegetable Images/train/Capsicum/*.jpg",
    "Vegetable Images/train/Carrot/*.jpg",
    "Vegetable Images/train/Cauliflower/*.jpg",
    "Vegetable Images/train/Cucumber/*.jpg",
    "Vegetable Images/train/Papaya/*.jpg",
    "Vegetable Images/train/Potato/*.jpg",
    "Vegetable Images/train/Pumpkin/*.jpg",
    "Vegetable Images/train/Radish/*.jpg",
    "Vegetable Images/train/Tomato/*.jpg"
]

image_all = []
labels = []
gray=[]
no_of_image = 5

for path in image_dataset:
    image_files = glob.iglob(path)
    count = 0

    for file in image_files:
        if count >= no_of_image:
            break

        image = cv2.imread(file)
        cv2.imshow("img",image)
        cv2.waitKey(0)
        print(image.shape)
        # Create label from folder name
        if "Bean" in path:
            label = "Bean"
        elif "Bitter_Gourd" in path:
            label = "Bitter_Gourd"
        elif "Bottle_Gourd" in path:
            label = "Bottle_Gourd"
        elif "Brinjal" in path:
            label = "Brinjal"
        elif "Broccoli" in path:
            label = "Broccoli"
        elif "Cabbage" in path:
            label = "Cabbage"
        elif "Capsicum" in path:
            label = "Capsicum"
        elif "Carrot" in path:
            label = "Carrot"
        elif "Cauliflower" in path:
            label = "Cauliflower"
        elif "Cucumber" in path:
            label = "Cucumber"
        elif "Papaya" in path:
            label = "Papaya"
        elif "Potato" in path:
            label = "Potato"
        elif "Pumpkin" in path:
            label = "Pumpkin"
        elif "Radish" in path:
            label = "Radish"
        elif "Tomato" in path:
            label = "Tomato"
    

        image_resized = cv2.resize(image, (256, 256))  # smaller size for faster training
        image_all.append(image_resized)

        gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        gray.append(gray_image)

        
        labels.append(label)
        count =count+1

print("Total images loaded:", len(image_all))
print("lemngth of labels",len(labels))
print(labels)

X = np.array(gray)
y = np.array(labels)
print(X)
# Normalize pixel values (0–255 → 0–1)
X = X / 255.0
print(X)
# Encode class labels into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(y_encoded)
y_encoded = to_categorical(y_encoded)
print(y_encoded)

model_path="model1.keras"
if os.path.exists(model_path):
    best_model=load_model(model_path)
    with open("test_data.pkl","rb") as f:
        X_test,y_test=pickle.load(f)
    y_pred=best_model.predict(X_test)
    print(y_pred)
    
else:


# Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
   
    with open("train_data.pkl","wb") as f:
        pickle.dump((X_train,y_train),f)

    with open("test_data.pkl","wb") as f:
        pickle.dump((X_test,y_test),f)


    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 1)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(15, activation='softmax')
    ])

    model=Sequential([
        SimpleRNN(32,(3,3),activation="relu",input_shape=(256,256,1)),
        Flatten(),
        Dense(64,activation="relu"),
        Dense(15,activation="softmax")

    ])


    model.compile(optimizer= Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    EPOCHS=6




    checkpoint_filepath="model1.keras"

    from tensorflow.keras.callbacks import ModelCheckpoint

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True
    )


    start_time = time.time()

    model.fit(X_train, y_train,epochs=EPOCHS,batch_size=1,validation_data=(X_test,y_test),callbacks=[model_checkpoint_callback])
    bestmodel=load_model(checkpoint_filepath)

    preds=bestmodel.predict(X_test)
    print(preds)
    preds=np.argmax(preds)
    print(preds)




    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test Accuracy",test_acc)


    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(y_pred_classes)
    y_true_classes = np.argmax(y_test, axis=1)
    print(y_true_classes)
    print("Predicted class indices:", y_pred_classes)
    print("True class indices:", y_true_classes)

