import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
nifti_file_path = "/data1/yuhan/mri/Lotus/output/infer-maisi-onestep-results/aab028_T1_encoded_onestep_t1.nii.gz"
# Select the axis and slice index for visualization (0=Sagittal, 1=Coronal, 2=Axial)
vis_axis = 2 # Axial view
slice_index = None # Set to None to automatically select the middle slice

# --- Load NIfTI File ---
if not os.path.exists(nifti_file_path):
    print(f"Error: File not found at {nifti_file_path}")
else:
    try:
        nii_img = nib.load(nifti_file_path)
        # Get image data as numpy array (usually float64 by default from get_fdata)
        img_data = nii_img.get_fdata()
        print(f"Successfully loaded image data with shape: {img_data.shape} and dtype: {img_data.dtype}")

        # --- Calculate Statistics ---
        mean_val = np.mean(img_data)
        std_val = np.std(img_data)
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        median_val = np.median(img_data)

        print("\n--- Image Statistics ---")
        print(f"Mean:      {mean_val:.4f}")
        print(f"Median:    {median_val:.4f}")
        print(f"Std Dev:   {std_val:.4f}")
        print(f"Min:       {min_val:.4f}")
        print(f"Max:       {max_val:.4f}")
        print(f"Dimensions: {img_data.shape}")
        # Print header info if needed
        # print("\n--- Header Info ---")
        # print(nii_img.header)

        # --- Visualize Middle Slice ---
        if slice_index is None:
            # Automatically select the middle slice along the chosen axis
            slice_index = img_data.shape[vis_axis] // 2

        # Ensure slice index is valid
        if 0 <= slice_index < img_data.shape[vis_axis]:
            print(f"\nDisplaying slice {slice_index} along axis {vis_axis}...")
            # Select the slice using numpy slicing
            if vis_axis == 0:
                slice_data = img_data[slice_index, :, :]
            elif vis_axis == 1:
                slice_data = img_data[:, slice_index, :]
            else: # vis_axis == 2
                slice_data = img_data[:, :, slice_index]

            # Display the slice
            plt.figure(figsize=(8, 8))
            plt.imshow(slice_data.T, cmap='gray', origin='lower') # Transpose for standard medical imaging orientation
            axis_names = ["Sagittal", "Coronal", "Axial"]
            plt.title(f"{axis_names[vis_axis]} Slice {slice_index}\n{os.path.basename(nifti_file_path)}")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 0")
            plt.colorbar(label="Intensity")
            plt.show()
        else:
            print(f"Error: Slice index {slice_index} is out of bounds for axis {vis_axis} (shape: {img_data.shape})")

    except Exception as e:
        print(f"An error occurred: {e}")
