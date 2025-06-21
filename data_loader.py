import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import xmltodict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SignLanguageDataLoader:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.classes = ['hello', 'thanks', 'yes', 'no', 'iloveu', 'sad', 'happy']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def parse_xml_annotation(self, xml_path):
        try:
            with open(xml_path, 'r') as file:
                xml_content = file.read()
            
            annotation = xmltodict.parse(xml_content)
            
            filename = annotation['annotation']['filename']
            
            bndbox = annotation['annotation']['object']['bndbox']
            xmin = int(bndbox['xmin'])
            ymin = int(bndbox['ymin'])
            xmax = int(bndbox['xmax'])
            ymax = int(bndbox['ymax'])
            
            class_name = annotation['annotation']['object']['name']
            
            if class_name == 'iloveyou':
                class_name = 'iloveu'
            
            return {
                'filename': filename,
                'class_name': class_name,
                'bbox': [xmin, ymin, xmax, ymax]
            }
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None
    
    def load_dataset(self, data_dir):
        images = []
        labels = []
        bboxes = []
        
        xml_files = [f for f in os.listdir(data_dir) if f.endswith('.xml')]
        print(f"Found {len(xml_files)} XML files in {data_dir}")
        
        successful_loads = 0
        failed_loads = 0
        
        for xml_file in xml_files:
            xml_path = os.path.join(data_dir, xml_file)
            annotation = self.parse_xml_annotation(xml_path)
            
            if annotation is None:
                failed_loads += 1
                continue
                
            img_path = os.path.join(data_dir, annotation['filename'])
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                failed_loads += 1
                continue
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not load image: {img_path}")
                    failed_loads += 1
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                xmin, ymin, xmax, ymax = annotation['bbox']
                
                h, w = img.shape[:2]
                xmin = max(0, min(xmin, w-1))
                ymin = max(0, min(ymin, h-1))
                xmax = max(xmin+1, min(xmax, w))
                ymax = max(ymin+1, min(ymax, h))
                
                if xmax <= xmin or ymax <= ymin:
                    print(f"Invalid bounding box for {img_path}: {annotation['bbox']}")
                    failed_loads += 1
                    continue
                
                cropped_img = img[ymin:ymax, xmin:xmax]
                
                if cropped_img.size == 0:
                    print(f"Empty cropped image for {img_path}")
                    failed_loads += 1
                    continue
                
                cropped_img = cv2.resize(cropped_img, (224, 224))
                
                cropped_img = cropped_img.astype(np.float32) / 255.0
                
                if annotation['class_name'] not in self.class_to_idx:
                    print(f"Unknown class '{annotation['class_name']}' in {img_path}")
                    failed_loads += 1
                    continue
                
                images.append(cropped_img)
                labels.append(self.class_to_idx[annotation['class_name']])
                bboxes.append(annotation['bbox'])
                successful_loads += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                failed_loads += 1
                continue
        
        print(f"Successfully loaded: {successful_loads} images")
        print(f"Failed to load: {failed_loads} images")
        
        assert len(images) == len(labels), f"Mismatch: {len(images)} images vs {len(labels)} labels"
        
        return np.array(images), np.array(labels), bboxes
    
    def load_train_test_data(self):
        print("Loading training data...")
        train_images, train_labels, train_bboxes = self.load_dataset(self.train_dir)
        
        print("Loading test data...")
        test_images, test_labels, test_bboxes = self.load_dataset(self.test_dir)
        
        print(f"Training samples: {len(train_images)}")
        print(f"Test samples: {len(test_images)}")
        print(f"Classes: {self.classes}")
        
        print("\nTraining class distribution:")
        for i, class_name in enumerate(self.classes):
            count = np.sum(train_labels == i)
            print(f"  {class_name}: {count}")
        
        print("\nTest class distribution:")
        for i, class_name in enumerate(self.classes):
            count = np.sum(test_labels == i)
            print(f"  {class_name}: {count}")
        
        return train_images, train_labels, test_images, test_labels
    
    def visualize_samples(self, images, labels, num_samples=5):
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        for i in range(num_samples):
            if i < len(images):
                axes[i].imshow(images[i])
                axes[i].set_title(f"Class: {self.classes[labels[i]]}")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    data_loader = SignLanguageDataLoader("train", "test")
    train_images, train_labels, test_images, test_labels = data_loader.load_train_test_data()
    
    data_loader.visualize_samples(train_images, train_labels) 