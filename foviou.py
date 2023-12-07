import math

# Função para converter graus em radianos
def deg_to_rad(degrees):
    return [math.radians(degree) for degree in degrees]

# Função para calcular o FoV-IoU
def fov_iou(Bg, Bd):
    θg, φg, αg, βg = Bg  # Desempacotando os valores da bounding box da verdade do terreno
    θd, φd, αd, βd = Bd  # Desempacotando os valores da bounding box detectada

    # Passo 1: Calcular a Área do FoV de Bg e Bd
    A_Bg = αg * βg
    A_Bd = αd * βd

    # Passo 2: Calcular a Distância do FoV entre Bg e Bd
    Δfov = (θd - θg) * math.cos((φg + φd) / 2)

    # Passo 3: Construir uma Intersecção Aproximada do FoV
    θI_min = max(-αg / 2, Δfov - αd / 2)
    θI_max = min(αg / 2, Δfov + αd / 2)
    φI_min = max(φg - βg / 2, φd - βd / 2)
    φI_max = min(φg + βg / 2, φd + βd / 2)

    # Passo 4: Calcular a Área da Intersecção e da União do FoV
    A_I = (θI_max - θI_min) * (φI_max - φI_min)
    A_U = A_Bg + A_Bd - A_I

    # Passo 5: Calcular o FoV-IoU
    FoV_IoU = A_I / A_U

    return FoV_IoU

# Exemplo de uso
b1 = deg_to_rad([40, 50, 35, 55])  # Bg
b2 = deg_to_rad([35, 20, 37, 50])  # Bd

print(b1, b2)

fov_iou_result = fov_iou(b1, b2)
print(fov_iou_result)