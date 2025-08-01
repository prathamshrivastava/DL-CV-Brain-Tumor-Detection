Brain Tumor Detection Using Deep Learning (VGG16 + Flask)

This project is a deep learning-based web application for detecting brain tumors from MRI images. It uses a **pre-trained VGG16** model fine-tuned on a custom dataset and provides a **Flask-powered web interface** for users to upload an image and receive predictions.

Classifies MRI brain scans into:
  - **Glioma**
  - **Meningioma**
  - **Pituitary**
  - **No Tumor**

Built on top of VGG16 (transfer learning)
Includes data augmentation, training history plots, and model evaluation
Simple web interface using Flask
Real-time predictions with confidence scores

Results
Class	Precision	Recall	F1-Score
Glioma	0.99	0.99	0.99
Meningioma	0.91	0.89	0.90
Pituitary	0.91	0.99	0.94
No Tumor	0.95	0.89	0.92
Accuracy			94%

Dataset Used
MRI images of brain tumors and healthy subjects were used from the Kaggle dataset (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

Future Improvements
-Improve UI styling and responsiveness
-Add support for multiple image uploads
-Deploy using Docker / on cloud (e.g., Render, Heroku, etc.)
-Model ensembling or upgrade to EfficientNet

Author
Pratham Shrivastava
Feel free to connect or suggest improvements!


