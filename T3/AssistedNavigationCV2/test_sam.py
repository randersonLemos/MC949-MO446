import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- Configuration ---
VIDEO_PATH = 'videos/bicicleta.mp4'         # <<<<< CHANGE THIS TO YOUR INPUT VIDEO FILE
OUTPUT_PATH = 'output_segmented.mp4'   # <<<<< CHANGE THIS TO YOUR DESIRED OUTPUT FILE
SAM_CHECKPOINT = 'sam_vit_h_4b8939.pth' # <<<<< CHANGE THIS TO YOUR DOWNLOADED CHECKPOINT FILE
MODEL_TYPE = 'vit_h'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# --- 1. Model Initialization ---
try:
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    
    # SamAutomaticMaskGenerator finds ALL objects in the image without a specific prompt
    mask_generator = SamAutomaticMaskGenerator(sam)
except Exception as e:
    print(f"Error initializing SAM model. Check your SAM_CHECKPOINT path and Model Type.")
    print(f"Details: {e}")
    exit()

# --- 2. Video Processing Setup (OpenCV) ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

## Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For MP4 file
#out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# --- 3. Frame-by-Frame Segmentation Loop ---
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}...")
    
    # SAM expects RGB format
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Generate all masks for the current frame
    masks = mask_generator.generate(image_rgb)
    
    # --- 4. Mask Visualization ---
    # Create an empty black image to draw the masks onto
    annotated_frame = np.zeros_like(frame)
    
    # Loop through all detected masks and color them randomly
    if masks:
        for i, mask_data in enumerate(masks):
            # Get the binary mask
            mask = mask_data['segmentation'] 
            
            # Create a random color for visualization
            color = np.random.randint(0, 256, 3) 
            
            # Apply color to the segmented area
            # We use an alpha blend to overlay the mask on the original frame
            for c in range(3):
                annotated_frame[:, :, c] = np.where(mask == 1, 
                                                    annotated_frame[:, :, c] * 0.5 + color[c] * 0.5, 
                                                    annotated_frame[:, :, c])
    
    # Blend the original frame with the mask overlay
    # This just visualizes the detected segments
    final_frame = cv2.addWeighted(frame, 0.5, annotated_frame, 0.5, 0)
    
    cv2.imshow('SAM', final_frame)
    cv2.waitKey(1)
    
    # --- 5. Save Frame ---
    #out.write(final_frame)

# --- 6. Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video segmentation complete. Output saved to {OUTPUT_PATH}")
