# ResNet50 CIFAR-10 Classification

This notebook demonstrates transfer learning for image classification on the CIFAR-10 dataset using a pre-trained ResNet50 model in TensorFlow/Keras.

## Features

- Loads and preprocesses CIFAR-10 data
- Data augmentation and resizing to 224x224 for ResNet50
- Uses ResNet50 (pre-trained on ImageNet) as a feature extractor
- Custom classification head with dropout and dense layers
- Model training with validation split and checkpointing
- Evaluation on test data
- Visualization of training loss and accuracy
- Confusion matrix and classification report

## Usage

1. **Install dependencies**  
   Make sure you have TensorFlow, scikit-learn, matplotlib, seaborn, and numpy installed.

   ```sh
   pip install tensorflow scikit-learn matplotlib seaborn numpy
   ```

2. **Run the notebook**  
   Open `Code1_(ResNet50).ipynb` in Jupyter Notebook or VS Code and run all cells.

## Main Steps

- **Data Loading:**  
  Loads CIFAR-10 and splits into training, validation, and test sets.

- **Data Augmentation & Resizing:**  
  Uses `ImageDataGenerator` and resizes images to 224x224 for ResNet50.

- **Model Definition:**  
  Loads ResNet50 without top layers, freezes base, adds custom head.

- **Training:**  
  Trains the model for 10 epochs, saving the best weights.

- **Evaluation:**  
  Loads best weights, evaluates on test set, and prints accuracy.

- **Visualization:**  
  Plots training loss and accuracy curves.

- **Analysis:**  
  Computes and plots confusion matrix, prints classification report.

## Output

- **Accuracy on training and test sets**
- **Loss and accuracy plots**
- **Confusion matrix heatmap**
- **Classification report (precision, recall, f1-score)**

## File Structure

- [`Code1_(ResNet50).ipynb`](Code1_(ResNet50).ipynb): Main notebook

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Keras ResNet50 Documentation](https://keras.io/api/applications/resnet/#resnet50-function)

👨‍💻 Author Muhammad Muaaz