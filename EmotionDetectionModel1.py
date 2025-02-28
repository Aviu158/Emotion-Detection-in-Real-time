# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:15:54 2024

@author: AdityaYadav
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import cv2
from keras.models import model_from_json
import numpy as np

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
	"D:/Emotion/train1",
	target_size=(48, 48),
	batch_size=32,
	color_mode="grayscale",
	class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
		"D:/Emotion/test",
		target_size=(48, 48),
		batch_size=32,
		color_mode="grayscale",
		class_mode='categorical')

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
						input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(5, activation='softmax'))

emotion_model.summary()

cv2.ocl.setUseOpenCL(False)

initial_learning_rate = 0.0001
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, 
							decay_rate=0.96)

optimizer = Adam(learning_rate=lr_schedule)

emotion_model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
					metrics=['accuracy'])

emotion_model_info = emotion_model.fit(
		train_generator,
		steps_per_epoch= 1278 // 8,
		epochs=30,
		validation_data=validation_generator,
		validation_steps=1105 // 8)

emotion_model.evaluate(validation_generator)

accuracy = emotion_model_info.history['accuracy']
val_accuracy = emotion_model_info.history['val_accuracy']
loss = emotion_model_info.history['loss']
val_loss = emotion_model_info.history['val_loss']


import matplotlib.pyplot as plt

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='accuracy')
plt.plot(val_accuracy, label='val accuracy')
plt.title('Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# loss graph
plt.subplot(1, 2, 2)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val loss')
plt.title('Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
	json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.weights.h5')

emotion_dict = {0: "Angry", 1: "Surprised", 2: "Sad", 
 				3: "Happy", 4: "Neutral"}

json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

cap = cv2.VideoCapture(0)
while True:
	# Find haar cascade to draw bounding box around face
	ret, frame = cap.read()
	frame = cv2.resize(frame, (1280, 720))
	if not ret:
		print(ret)
	# Create a face detector
	face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces available on camera
	num_faces = face_detector.detectMultiScale(gray_frame, 
											scaleFactor=1.3, minNeighbors=5)

	# take each face available on the camera and Preprocess it
	for (x, y, w, h) in num_faces:
        
	    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
	    roi_gray_frame = gray_frame[y:y + h, x:x + w]
	    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, 
															(48, 48)), -1), 0)

		# predict the emotions
	    emotion_prediction = emotion_model.predict(cropped_img)
	    maxindex = int(np.argmax(emotion_prediction))
	    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), 
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow('Emotion Detection', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

