from keras.utils import plot_model
from functions import *
import os
from pathlib import Path

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
EPOCHS = 5
CHUNK_SIZE = 1000




if __name__ == "__main__":
    train_generator, val_generator, test_generator = split_and_get_generators(IMAGE_SIZE, CHUNK_SIZE, BATCH_SIZE)
    
    print(f"Train samples: {train_generator.n_samples}")
    print(f"Val samples: {val_generator.n_samples}")
    
    # Build model
    model = build_model(IMAGE_SIZE)
    #model = load_model("final_model.keras")
    
    # Train with validation
    print("Starting training...")
    history = train_with_validation(model, train_generator, val_generator, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_model(model, test_generator)
    plot_training_history(history)
    
    # Save final model
    model.save("final_model.keras")
    print("Model saved as final_model.keras")