# CycleGAN-Implementation-for-Person-Face-Sketches
Implement the CycleGAN model based on the original paper, using the same Person Face Sketches dataset as provided in Question #3. The goal of this task is to convert a face image into a sketch and a sketch back into a real face (image-to-image translation).
At test time, the model should be able to:
-
Take a sketch and generate a corresponding real face image.
-
Take a real face image and generate a corresponding sketch.
The model will be trained end-to-end, and I highly recommend using Google Colab for this task. Make sure to save model weights after every epoch to avoid losing progress. If you need to restart the training for any reason, resume from the saved weights rather than starting from scratch.
If you encounter memory issues and cannot load the entire dataset at once, you can:
-
Split the dataset into smaller batches.
-
Reduce the number of training samples as a last resort.
Train your model for multiple epochs. Carefully monitor the training process and adjust hyperparameters (e.g., learning rate, batch size) as necessary to improve model performance.
