import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def compare_segmentation_models(models_dict, dataset, num_samples=3, shuffle=False, index_range=None, save_dir=None):
    """
    Plots Original Image, Ground Truth, and Predictions for multiple models side-by-side,
    with an option to save the plots locally.
    
    Args:
        models_dict (dict): Dictionary of models, e.g., {"Base CCE": model1, "Dice": model2}
        dataset (tf.data.Dataset): The batched evaluation/test dataset.
        num_samples (int): Number of images to visualize (used if index_range is None).
        shuffle (bool): If True, randomly shuffles the dataset to show random examples.
        index_range (tuple): (start_idx, end_idx) to view a specific slice of the dataset.
        save_dir (str): Local folder path where images should be saved. If None, images won't be saved.
    """
    # 1. Create the save directory if it doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Images will be saved to local folder: '{save_dir}'")

    ds = dataset.unbatch()
    
    # Base tracking index for unique filenames
    start_offset = 0
    
    if index_range is not None:
        start_idx, end_idx = index_range
        start_offset = start_idx
        samples_to_plot = ds.skip(start_idx).take(end_idx - start_idx)
        print(f"Showing dataset slice from index {start_idx} to {end_idx}...")
    elif shuffle:
        samples_to_plot = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True).take(num_samples)
        print(f"Showing {num_samples} random samples...")
    else:
        samples_to_plot = ds.take(num_samples)
        print(f"Showing the first {num_samples} samples...")

    custom_cmap = ListedColormap(["red", "green", "black"])
    model_names = list(models_dict.keys())
    num_models = len(models_dict)
    total_cols = 2 + num_models
    
    for i, (image, mask) in enumerate(samples_to_plot):
        fig, axes = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))
        axes = axes.flatten()
        
        img_np = image.numpy()
        mask_np = tf.squeeze(mask).numpy()
        
        if img_np.min() < 0:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
        # Determine unique ID for the filename
        sample_id = start_offset + i if index_range is not None else f"random_{i}"
            
        # Column 0: Original
        axes[0].imshow(img_np)
        axes[0].set_title(f"Sample {sample_id}\nOriginal Image", fontsize=10, fontweight='bold')
        axes[0].axis("off")
        
        # Column 1: Ground Truth
        axes[1].imshow(mask_np, cmap=custom_cmap)
        axes[1].set_title("Ground Truth", fontsize=10, fontweight='bold')
        axes[1].axis("off")
        
        # Columns 2+: Predictions
        input_tensor = tf.expand_dims(image, axis=0)
        for idx, name in enumerate(model_names):
            model = models_dict[name]
            pred = model.predict(input_tensor, verbose=0)
            pred_mask = np.argmax(pred, axis=-1)[0]
            
            col_idx = 2 + idx
            axes[col_idx].imshow(pred_mask, cmap=custom_cmap)
            axes[col_idx].set_title(f"{name}", fontsize=10, fontweight='bold')
            axes[col_idx].axis("off")
            
        plt.tight_layout()
        
        # --- Local Saving Mechanism ---
        if save_dir is not None:
            # Cleans up model names for file system safety and defines the output path
            filename = f"model_comparison_sample_{sample_id}.png"
            save_path = os.path.join(save_dir, filename)
            
            # dpi=300 ensures crisp, high-quality images for your project report/slides
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_path}")
            
        plt.show()


'''
compare_segmentation_models(
    models_dict=my_models,
    dataset=test_ds,
    num_samples=3,
    shuffle=True # <--- Activates random mode
    save_dir="results_plots"  # <--- Saves them inside a folder named 'results_plots'
)
compare_segmentation_models(
    models_dict=my_models,
    dataset=test_ds,
    index_range=(10, 15) # <--- Will show images 10, 11, 12, 13, and 14
)
'''