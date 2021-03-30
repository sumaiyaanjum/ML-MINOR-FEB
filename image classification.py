!pip install ipython-autotime
%load_ext autotime 
!pip install bing-image-downloader
!mkdir images
from bing_image_downloader import downloader
downloader.download("red roses",limit=30,output_dir='images',
                    adult_filter_off=True)
from bing_image_downloader import downloader
downloader.download("cars",limit=30,output_dir='images',
                    adult_filter_off=True)
from bing_image_downloader import downloader
downloader.download("nature",limit=30,output_dir='images',
                    adult_filter_off=True)
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize 

target = []
images = []
flat_data = []

DATADIR = '/content/images/bicycle'
CATEGORIES =['red roses','cars','nature']

for category in CATEGORIES:
  class_num = CATEGORIES.index(category)
  print(class_num)
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize 

target = []
images = []
flat_data = []

DATADIR = '/content/images'
CATEGORIES =['red roses','cars','nature']

for category in CATEGORIES:
  class_num = CATEGORIES.index(category)
  path = os.path.join(DATADIR,category)
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img))
    print(img_array)
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize 

target = []
images = []
flat_data = []

DATADIR = '/content/images'
CATEGORIES =['red roses','cars','nature']

for category in CATEGORIES:
  class_num = CATEGORIES.index(category)
  path = os.path.join(DATADIR,category)
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img))
    print(img_array.shape)
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize 

target = []
images = []
flat_data = []

DATADIR = '/content/images'
CATEGORIES =['red roses','cars','nature']

for category in CATEGORIES:
  class_num = CATEGORIES.index(category)
  path = os.path.join(DATADIR,category)
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img))
    plt.imshow(img_array)
    break
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize 

target = []
images = []
flat_data = []

DATADIR = '/content/images'
CATEGORIES =['red roses','cars','nature']

for category in CATEGORIES:
  class_num = CATEGORIES.index(category)
  path = os.path.join(DATADIR,category)
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img)) #normalise the values from 0 to 1
    img_resized = resize(img_array,(130,130,3))
    flat_data.append(img_resized.flatten())
    images.append(img_resized)
    target.append(class_num)

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)
flat_data[0]
flat_data[3]

130*130*3
len(flat_data[3])
target
unique,count = np.unique(target,return_counts=True)
plt.bar(CATEGORIES,count)
# splitting the data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(flat_data,target,
                                  test_size=0.5,random_state=112)
from sklearn.model_selection import GridSearchCV
from sklearn import svm
param_grid = [
              {'C':[1,10,100,1000],'kernel':['linear']},
              {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']}
]

svc = svm.SVC(probability=True)
clf = GridSearchCV(svc,param_grid)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
y_pred
y_test
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_pred,y_test)
confusion_matrix(y_pred,y_test)
import pickle
pickle.dump(clf,open('img_model.p','wb'))
#testing new image
flat_data = []
url = input('Enter your URL')
img = imread(url)
img_resized = resize(img,(130,130,3))
flat_data.append(img_resized.flatten()) 
flat_data = np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out = model.predict(flat_data)
y_out = CATEGORIES[y_out[0]]
print(f' PREDICTED OUTPUT: {y_out}')
!pip install streamlit

!pip install pyngrok
from pyngrok import  ngrok
