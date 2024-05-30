Image Classifier with Gradio Interface:
This repository contains an image classification application built with Gradio, PIL, and Keras. The application uses a pre-trained Keras model to classify images as either "Cat" or "Dog."

Features:
Simple Interface: Easy-to-use Gradio interface for image upload and classification.
Pre-trained Model: Uses a Keras model trained on a dataset of cats and dogs.
Real-time Prediction: Provides instant classification results with confidence scores.

Installation:
1. Clone the repository:
(git clone https://github.com/vkulkarni33/image-classifier.git)
(cd image-classifier)

2. Install the required dependencies:
pip install -r requirements.txt

3. Make sure you have the following files in the /content directory:
keras_model.h5: The pre-trained Keras model.
labels.txt: A text file containing the class names.

Code Explanation:

Loading the Model and Labels:
The Keras model and class names are loaded at the beginning:
model = load_model("/content/keras_model.h5", compile=False)
class_names = open("/content/labels.txt", "r").readlines()

Image Classification Function:
The classify_image function processes the input image and returns the classification result:
def classify_image(image):
    
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    result_text = f"Class: {class_name}\nConfidence Score: {confidence_score:.2f}"
    return result_text


Gradio Interface:
The Gradio interface is created and launched using the gr.Interface function:


    interface = gr.Interface(
     fn=classify_image,
     inputs=gr.Image(),
     outputs="text",
     title="Image Classifier",
     description="Upload an image to classify it as a Cat or Dog."
    )
    interface.launch()


Contributing.
Contributions are welcome! Please fork the repository and create a pull request with your changes.

This README provides a comprehensive guide to setting up and using the image classifier application. Feel free to customize it further to match your project's specifics and style.













