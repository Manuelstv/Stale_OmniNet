def pixel_to_spherical(x, y, image_width, image_height, HFOV, VFOV):
    # Normalize pixel coordinates to [-1, 1]
    x_normalized = (x / (image_width / 2)) - 1
    y_normalized = (y / (image_height / 2)) - 1

    # Convert normalized coordinates to spherical coordinates
    theta = x_normalized * HFOV / 2
    phi = y_normalized * VFOV / 2

    return theta, phi

# Assume a bounding box in pixel coordinates
#bbox_pixel_coords = [x_c, y_c, alpha, beta]
bbox_pixel_coords = [0, 300,4,4]

# Image dimensions and camera FoV
image_width = 600  # Example width
image_height = 300  # Example height
HFOV = 360  # Example HFOV for a 360° camera
VFOV = 180  # Example VFOV for a 360° camera

# Calculate center coordinates in spherical format
theta_center, phi_center = pixel_to_spherical(
    bbox_pixel_coords[0],
    bbox_pixel_coords[1],
    image_width, image_height, HFOV, VFOV
)

# The final FoV-BB format
print((theta_center, phi_center))
