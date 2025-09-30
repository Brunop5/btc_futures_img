from functions import *
import os
from pathlib import Path

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 10
CHUNK_SIZE = 1000




if __name__ == "__main__":
    # Check if data is already preprocessed
    data_dir = Path("picture_data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    train_files = list(train_dir.glob("*.npz"))
     
    if not train_files:
        print("Preprocessing data...")
        data = pd.read_csv("1h_test.csv", index_col=0, parse_dates=True)
        train_data = data
        
        # Split data first
        train_chunk, temp_chunk = train_test_split(train_data, test_size=0.3, random_state=42, shuffle=False)
        val_chunk, test_chunk = train_test_split(temp_chunk, test_size=0.5, random_state=42, shuffle=False)
        
        print(f"Train: {len(train_chunk)}, Val: {len(val_chunk)}, Test: {len(test_chunk)}")
        
        # Preprocess each split separately
        preprocess_and_save_chunks(train_chunk, chunk_size=CHUNK_SIZE, 
                                  image_size=IMAGE_SIZE, data_dir="picture_data/train")
        preprocess_and_save_chunks(val_chunk, chunk_size=CHUNK_SIZE, 
                                  image_size=IMAGE_SIZE, data_dir="picture_data/val")
        preprocess_and_save_chunks(test_chunk, chunk_size=CHUNK_SIZE, 
                                  image_size=IMAGE_SIZE, data_dir="picture_data/test")
    else:
        print(f"Found {len(train_files)} existing train files, skipping preprocessing")
    
    # Create generators for each split
    train_generator = ImageDataGenerator(data_dir="picture_data/train", 
                                       batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    val_generator = ImageDataGenerator(data_dir="picture_data/val", 
                                     batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    test_generator = ImageDataGenerator(data_dir="picture_data/test", 
                                     batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    
    print(f"Train samples: {train_generator.n_samples}")
    print(f"Val samples: {val_generator.n_samples}")
    
    # Build model
    model = build_model(IMAGE_SIZE)
    #model = load_model("final_model.keras")
    
    # Train with validation
    print("Starting training...")
    train_with_validation(model, train_generator, val_generator, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_model(model, test_generator)
    
    # Save final model
    model.save("final_model.keras")
    print("Model saved as final_model.keras")