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

    return max(FoV_IoU,0)


# Exemplo de uso
b1 = deg_to_rad([40, 50, 35, 55, 0])  # Bg
b2 = deg_to_rad([35, 20, 37, 50, 0])  # Bd

print(b1, b2)

fov_iou_result = fov_iou(b1, b2)
print(fov_iou_result)