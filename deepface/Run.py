from deepface import DeepFace
import tensorflow as tf
import cv2

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
DeepFace.stream(db_path=r"./images", model_name='Facenet', detector_backend='opencv',
                source=0, enable_face_analysis=False, time_threshold=1, frame_threshold=1)
