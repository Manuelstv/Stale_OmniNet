import math
import torch
from pdb import set_trace as pause
import numpy as np



def angle2radian(angle_sph_box, mode='convention'):
    """
    Transform angle to radian based on specific mode.

    Args:
        angle_sph_box (list or list of lists): Box or list of boxes with angle representation.
        mode (str, optional): Mode for angle conversion. Defaults to 'convention'.
            'convention': (90, -90), (-180, 180)
            'math': (0, 180), (0, 360)

    Returns:
        radian_sph_box (list or list of lists): Box or list of boxes with radian representation.
    """
    assert mode in ['math', 'convention']

    # Check if the input is a single list (1D) or a list of lists (2D)
    if isinstance(angle_sph_box[0], list):
        # Input is a list of lists (2D)
        radian_sph_box = [[math.radians(angle) for angle in box] for box in angle_sph_box]
    else:
        # Input is a single list (1D)
        radian_sph_box = [math.radians(angle) for angle in angle_sph_box]

    if mode == 'convention':
        if isinstance(radian_sph_box[0], list):
            # Process each box in the list of boxes
            for box in radian_sph_box:
                box[0] = box[0] - math.pi #0, 2pi -> -pi, pi
                box[1] = math.pi / 2 - box[1] # 0,pi -> -pi/2, pi/2 
        else:
            # Process the single list of angles
            radian_sph_box[0] = radian_sph_box[0] - math.pi
            radian_sph_box[1] = math.pi / 2 - radian_sph_box[1]

    return radian_sph_box

# Função para converter graus em radianos
def deg2rad(degrees):
    return [math.radians(degree) for degree in degrees]

def fov_iou(Bg, Bd):
    import math

    theta_g, phi_g, alpha_g, beta_g = Bg  # Unpacking ground truth bounding box values
    theta_d, phi_d, alpha_d, beta_d = Bd  # Unpacking detected bounding box values

    # Step 1: Calculate FoV Area of Bg and Bd
    A_Bg = alpha_g * beta_g
    A_Bd = alpha_d * beta_d

    # Step 2: Calculate FoV distance between Bg and Bd
    delta_fov = (theta_d - theta_g) * math.cos((phi_g + phi_d) / 2)

    # Step 3: Construct an approximate FoV Intersection
    theta_I_min = max(-alpha_g / 2, delta_fov - alpha_d / 2)
    theta_I_max = min(alpha_g / 2, delta_fov + alpha_d / 2)
    phi_I_min = max(phi_g - beta_g / 2, phi_d - beta_d / 2)
    phi_I_max = min(phi_g + beta_g / 2, phi_d + beta_d / 2)

    # Step 4: Calculate the Area of the FoV Intersection and Union
    if theta_I_max > theta_I_min and phi_I_max > phi_I_min:
        # Compute area of FoV intersection I
        A_I = (theta_I_max - theta_I_min) * (phi_I_max - phi_I_min)
    else:
        # No valid intersection
        A_I = 0

    A_U = A_Bg + A_Bd - A_I

    # Ensure that A_I and A_U are not negative
    #A_U = max(A_U, 0)

    # Step 5: Calculate the FoV-IoU
    FoV_IoU = A_I / A_U if A_U != 0 else 0

    if FoV_IoU>1:
        print(Bg, Bd)

    return FoV_IoU

def fov_giou_loss(Bg, Bd):
    # Bg and Bd are the ground truth and detected bounding boxes
    theta_g, phi_g, alpha_g, beta_g = Bg  # Unpacking ground truth bounding box values
    theta_d, phi_d, alpha_d, beta_d = Bd  # Unpacking detected bounding box values
    
    # Step 1: Compute Δfov, FoV intersection I and FoV union U as Algorithm 1
    delta_fov = (Bd[0] - Bg[0]) * torch.cos((Bg[1] + Bd[1]) / 2)
    theta_min = torch.max(-Bg[2]/2, -alpha_d/2 - delta_fov)
    theta_max = torch.min(Bg[2]/2, alpha_d/2 + delta_fov)
    phi_min = torch.max(-Bg[3]/2, -Bd[3]/2 - delta_fov)
    phi_max = torch.min(Bg[3]/2, Bd[3]/2 + delta_fov)
    A_I = (theta_max - theta_min) * (phi_max - phi_min)
    A_U = Bg[2] * Bg[3] + Bd[2] * Bd[3] - A_I
    
    # Step 2: Build an approximate smallest enclosing box C
    theta_c_min = torch.min(-Bg[2]/2, -Bd[2]/2 - delta_fov)
    theta_c_max = torch.max(Bg[2]/2, Bd[2]/2 + delta_fov)
    phi_c_min = torch.min(-Bg[3]/2, -Bd[3]/2 - delta_fov)
    phi_c_max = torch.max(Bg[3]/2, Bd[3]/2 + delta_fov)

    # Step 3: Compute area of smallest enclosing box C
    A_C = (theta_c_max - theta_c_min) * (phi_c_max - phi_c_min)
    
    # Step 4: Compute FoV-GIoU loss
    FoV_GIoU = 1 - (A_I / A_U) + (A_C - A_U) / A_C
    
    return FoV_GIoU

# Function to wrap a value within a specified range
def wrap_within_range(value, value_range):
    lower, upper = value_range
    while value < lower:
        value += (upper - lower)
    while value > upper:
        value -= (upper - lower)
    return value

# Function to translate spherical coordinates
def translate_coordinates(coordinates, origin_deg):
    translated_coordinates_deg = []
    for coord in coordinates:
        theta_deg = wrap_within_range(coord[0] - origin_deg[0], theta_range_deg)
        phi_deg = wrap_within_range(coord[1] - origin_deg[1], phi_range_deg)
        #fov_theta_deg = wrap_within_range(coord[2], fov_theta_range_deg)
        #fov_phi_deg = wrap_within_range(coord[3], fov_phi_range_deg)
        translated_coordinates_deg.append([theta_deg, phi_deg, coord[2], coord[3]])
    return translated_coordinates_deg

'''
b1 = [30, 75, 30, 30]
b2 = [60, 55, 40, 50, 0]


b1 = [40, 70, 25, 30]
b2 = [60, 85, 30, 30]

b1 = [30, 75, 30, 60]
b2 = [60, 40, 60, 60]

theta_origin_deg = 0
phi_origin_deg = 0

# Define the ranges for each coordinate component
theta_range_deg = (-180, 180)
phi_range_deg = (-90, 90)
fov_theta_range_deg = (0, 90)
fov_phi_range_deg = (0, 90)

# Translate b1 and b2 coordinates
translated_b1 = translate_coordinates([b1], [theta_origin_deg, phi_origin_deg])[0]
translated_b2 = translate_coordinates([b2], [theta_origin_deg, phi_origin_deg])[0]
'''

if __name__ == '__main__':
# Test case from Table I
    b1 = [0,   54,   24,   20]  # BFoV parameters for Bg
    b2 = [150, -2, 45, 45]  # BFoV parameters for Bd

    '''
    theta_origin_deg = 0
    phi_origin_deg = 0

    # Define the ranges for each coordinate component
    theta_range_deg = (-180, 180)
    phi_range_deg = (-90, 90)

    # Translate b1 and b2 coordinates
    b1 = translate_coordinates([b1], [theta_origin_deg, phi_origin_deg])[0]
    b2 = translate_coordinates([b2], [theta_origin_deg, phi_origin_deg])[0]

    print(b1, b2)'''

    #b1 = [30, 60, 60, 60]
    #b2 = [60, 60, 60, 60]

    b1_rad = deg2rad(b1)
    b2_rad = deg2rad(b2)

    fov_iou_result = fov_iou(b1_rad, b2_rad)
    print(fov_iou_result)  # Prints FoV IoU for b1 and b2

    # Additional test cases from Table II
    # Table II.a
    b1 = [40, 50, 35, 55]
    b2 = [35, 20, 37, 50]
    b1_rad = angle2radian(b1)
    b2_rad = angle2radian(b2)

    print(fov_iou(b1_rad, b2_rad))

    # Table II.b
    b1 = [30, 60, 60, 60]
    b2 = [55, 40, 60, 60]
    b1_rad = angle2radian(b1)
    b2_rad = angle2radian(b2)

    b1_rad = angle2radian(b1)
    b2_rad = angle2radian(b2)

    print(fov_iou(b1_rad, b2_rad))

    # Table II.c
    b1 = [50, -78, 25, 46]
    b2 = [30, -75, 26, 45]
    b1_rad = angle2radian(b1)
    b2_rad = angle2radian(b2)
    print(fov_iou(b1_rad, b2_rad))

    # Table II.d
    b1 = [30, 75, 30, 60]
    b2 = [60, 40, 60, 60]
    b1_rad = angle2radian(b1)
    b2_rad = angle2radian(b2)
    print(fov_iou(b1_rad, b2_rad))

    b1 = [40, 70, 25, 30]
    b2 = [60, 85, 30, 30]
    b1_rad = angle2radian(b1)
    b2_rad = angle2radian(b2)
    print(fov_iou(b1_rad, b2_rad))

    b1 = [30, 75, 30, 30]
    b2 = [60, 55, 40, 50]
    b1_rad = angle2radian(b1)
    b2_rad = angle2radian(b2)
    print(fov_iou(b1_rad, b2_rad))