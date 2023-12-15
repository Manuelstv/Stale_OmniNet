import math

# Função para converter graus em radianos
def deg_to_rad(degrees):
    return [math.radians(degree) for degree in degrees]

def fov_iou(Bg, Bd):
    theta_g, phi_g, alpha_g, beta_g, _ = Bg  # Desempacotando os valores da bounding box da verdade do terreno
    theta_d, phi_d, alpha_d, beta_d, _ = Bd  # Desempacotando os valores da bounding box detectada

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

    FoV_IoU = max(FoV_IoU,0)

    if FoV_IoU>=1:
        FoV_IoU=0
        #pass
    
    return FoV_IoU



b1 = [0, 12, 28, 24,  0]
b2 = [0, 12, 28, 24,  0]

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
        translated_coordinates_deg.append([theta_deg, phi_deg, fov_theta_deg, fov_phi_deg,0])
    return translated_coordinates_deg

# Translate b1 and b2 coordinates
translated_b1 = translate_coordinates([b1], [theta_origin_deg, phi_origin_deg])[0]
translated_b2 = translate_coordinates([b2], [theta_origin_deg, phi_origin_deg])[0]

print(translated_b1)
print(translated_b2)

#b1 = [-78.0, -80.0, 45.0, 30.0,0]
#b2 = [-111.0, -18.0, 5.0, 51,0]

#b1 = [-159.0, 9.0, 45.0, 30.0,0]
#b2 = [169.0, 72.0, 65.0, 51.0,0]
# Exemplo de uso

#translated_b2 = [147, 62,65,51,0]
b1 = deg_to_rad(translated_b1)  # Bg
b2 = deg_to_rad(translated_b2)  # Bd

#b1 = [-0.4014257279586958*180, -0.4188790204786391*90, 0.13962634015954636*90, 0.6283185307179586*90, 0.0]
#b2 = [1.4660765716752369, 0.9948376736367679, 0.8377580409572782, 1.2217304763960306, 0.0]

fov_iou_result = fov_iou(b1, b2)
print(fov_iou_result)