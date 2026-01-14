import cv2
import glob
import matplotlib.pyplot as plt
image_dataset=["Vegetable Images\\train\Bean\\*jpg","Vegetable Images\train\Bitter_Gourd\\*jpg",]

train_images = []
GREY_IMAGE=[]
cout=0
for img_path in glob.glob("Vegetable Images/train/*/*.jpg"):
    
    img = cv2.imread(img_path)
    grey_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hsv_image=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    rgb_image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title("original image")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(grey_image,cmap="gray")
    plt.title("grey image")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(hsv_image)
    plt.title("hsv image")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(rgb_image)
    plt.title("rgb image")
    plt.axis("off")

    plt.suptitle("color space model")
    plt.show()



    train_images.append(img)
    GREY_IMAGE.append(grey_image)
    if cout==5:
        break
    cout=cout+1
    

print("Total train images:", len(train_images))
print(train_images[0])

for i in GREY_IMAGE:
    plt.imshow(i)
    plt.show()


