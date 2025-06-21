import os
import matplotlib.pyplot as plt
import numpy as np
from data_loader import SignLanguageDataLoader
from model import SignLanguageModel

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_performance(model, test_images, test_labels, data_loader):
    """
    Evaluate model performance with detailed metrics
    
    Args:
        model: Trained model
        test_images: Test images
        test_labels: Test labels
        data_loader: Data loader instance
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Get predictions
    predictions = model.model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels, 
                              target_names=data_loader.classes))
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data_loader.classes,
                yticklabels=data_loader.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training function
    """
    print("=== Sign Language Recognition Model Training ===")
    
    # Load data
    print("\n1. Loading training data...")
    data_loader = SignLanguageDataLoader("train", "test")
    train_images, train_labels, test_images, test_labels = data_loader.load_train_test_data()
    
    if len(train_images) == 0 or len(test_images) == 0:
        print("Error: No training or test data found!")
        print("Please make sure you have images and XML files in the train/ and test/ directories.")
        return
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Classes: {data_loader.classes}")
    
    # Visualize some samples
    print("\n2. Visualizing sample images...")
    data_loader.visualize_samples(train_images, train_labels, num_samples=6)
    
    # Create and train model
    print("\n3. Creating and training model...")
    model = SignLanguageModel(num_classes=len(data_loader.classes))
    
    # Choose model architecture
    print("Choose model architecture:")
    print("1. Simple CNN (faster training, smaller model)")
    print("2. Transfer Learning with MobileNetV2 (better accuracy, slower training)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        print("Building transfer learning model...")
        model.build_model()
    else:
        print("Building simple CNN model...")
        model.build_simple_cnn()
    
    # Display model summary
    print("\nModel Architecture:")
    model.model.summary()
    
    # Training parameters
    epochs = int(input("\nEnter number of training epochs (default: 30): ") or "30")
    batch_size = int(input("Enter batch size (default: 32): ") or "32")
    
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    
    # Train the model
    history = model.train(train_images, train_labels, test_images, test_labels, 
                         epochs=epochs, batch_size=batch_size)
    
    # Plot training history
    print("\n4. Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n5. Evaluating model performance...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    
    # Detailed evaluation
    evaluate_model_performance(model, test_images, test_labels, data_loader)
    
    # Save the model
    print("\n6. Saving model...")
    model_path = "sign_language_model.h5"
    model.save_model(model_path)
    
    print(f"\n=== Training Completed ===")
    print(f"Model saved as: {model_path}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Ask if user wants to run real-time detection
    run_detection = input("\nDo you want to run real-time detection now? (y/n): ").strip().lower()
    if run_detection == 'y':
        print("Starting real-time detection...")
        from real_time_detection import RealTimeSignLanguageDetector
        detector = RealTimeSignLanguageDetector(model_path)
        detector.run_detection()

if __name__ == "__main__":
    main() 