def pixel_to_spherical(x, y, image_width, image_height, HFOV, VFOV):
    # Normalize pixel coordinates to [-1, 1]
    x_normalized = (x / (image_width / 2)) - 1
    y_normalized = (y / (image_height / 2)) - 1

    # Convert normalized coordinates to spherical coordinates
    theta = x_normalized * HFOV / 2
    phi = y_normalized * VFOV / 2

    return theta, phi

# Assume a bounding box in pixel coordinates
bbox_pixel_coords = [x_min, y_min, x_max, y_max]

# Image dimensions and camera FoV
image_width = 1920  # Example width
image_height = 1080  # Example height
HFOV = 360  # Example HFOV for a 360° camera
VFOV = 180  # Example VFOV for a 360° camera

# Calculate center coordinates in spherical format
theta_center, phi_center = pixel_to_spherical(
    (bbox_pixel_coords[0] + bbox_pixel_coords[2]) / 2,
    (bbox_pixel_coords[1] + bbox_pixel_coords[3]) / 2,
    image_width, image_height, HFOV, VFOV
)

# Calculate alpha and beta based on the bounding box size and image dimensions
alpha = (bbox_pixel_coords[2] - bbox_pixel_coords[0]) / image_width * HFOV
beta = (bbox_pixel_coords[3] - bbox_pixel_coords[1]) / image_height * VFOV

# The final FoV-BB format
fov_bb = (theta_center, phi_center, alpha, beta)
