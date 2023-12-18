import math
import torch


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
                box[0] = box[0] - math.pi
                box[1] = math.pi / 2 - box[1]
        else:
            # Process the single list of angles
            radian_sph_box[0] = radian_sph_box[0] - math.pi
            radian_sph_box[1] = math.pi / 2 - radian_sph_box[1]

    return radian_sph_box

# Função para converter graus em radianos
def deg_to_rad(degrees):
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
    A_I = (theta_I_max - theta_I_min) * (phi_I_max - phi_I_min)
    A_U = A_Bg + A_Bd - A_I

    # Ensure that A_I and A_U are not negative
    A_I = max(A_I, 0)
    A_U = max(A_U, 0)

    # Step 5: Calculate the FoV-IoU
    FoV_IoU = A_I / A_U if A_U != 0 else 0

    return FoV_IoU

def fov_giou_loss(Bg, Bd):
    # Bg and Bd are the ground truth and detected bounding boxes
    # Bg = (θg, φg, αg, βg), Bd = (θd, φd, αd, βd)
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

def fov_iou_stale(Bg, Bd):
    theta_g, phi_g, alpha_g, beta_g = Bg  # Desempacotando os valores da bounding box da verdade do terreno
    theta_d, phi_d, alpha_d, beta_d = Bd  # Desempacotando os valores da bounding box detectada

    # Passo 1: Calcular a Área do FoV de Bg e Bd
    A_Bg = alpha_g * beta_g
    A_Bd = alpha_d * beta_d

    # Passo 2: Calcular a Distância do FoV entre Bg e Bd
    delta_fov = (theta_d - theta_g) * math.cos((phi_g + phi_d) / 2)

    # Passo 3: Construir uma Intersecção Aproximada do FoV
    theta_I_min = max(-alpha_g / 2, delta_fov - alpha_d / 2)
    theta_I_max = min(alpha_g / 2, delta_fov + alpha_d / 2)
    phi_I_min = max(phi_g - beta_g / 2, phi_d - beta_d / 2)
    phi_I_max = min(phi_g + beta_g / 2, phi_d + beta_d / 2)

    # Passo 4: Calcular a Área da Intersecção e da União do FoV
    A_I = (theta_I_max - theta_I_min) * (phi_I_max - phi_I_min)
    A_U = A_Bg + A_Bd - A_I

    # Passo 5: Calcular o FoV-IoU
    FoV_IoU = A_I / A_U

    #FoV_IoU = max(FoV_IoU,0)

    #if FoV_IoU>=1:
    #    FoV_IoU=0
        #pass
    
    return FoV_IoU

b1 = [30, 75, 30, 30]
b2 = [60, 55, 40, 50, 0]

#Wb1 = [-158.0, 10.0, 45.0, 30.0]
#b2 = [169.0, 72.0, 65.0, 51.0]

b1 = [40, 70, 25, 30]
b2 = [60, 85, 30, 30]

#b1 = [30, 75, 30, 60]
#b2 = [60, 40, 60, 60]

theta_origin_deg = 0
phi_origin_deg = 0

# Define the ranges for each coordinate component
theta_range_deg = (-180, 180)
phi_range_deg = (-90, 90)
fov_theta_range_deg = (0, 90)
fov_phi_range_deg = (0, 90)

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
        fov_theta_deg = wrap_within_range(coord[2], fov_theta_range_deg)
        fov_phi_deg = wrap_within_range(coord[3], fov_phi_range_deg)
        translated_coordinates_deg.append([theta_deg, phi_deg, fov_theta_deg, fov_phi_deg])
    return translated_coordinates_deg

# Translate b1 and b2 coordinates
translated_b1 = translate_coordinates([b1], [theta_origin_deg, phi_origin_deg])[0]
translated_b2 = translate_coordinates([b2], [theta_origin_deg, phi_origin_deg])[0]

b1 = angle2radian(translated_b1)
b2 = angle2radian(translated_b2)

print(b1)

fov_iou_result = fov_iou(b1, b2)
print(fov_iou_result)