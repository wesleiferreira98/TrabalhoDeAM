import imgaug as ia
import imgaug.augmenters as iaa
import cv2 
from glob import glob 

# Defina as transformações desejadas
seq = iaa.Sequential([
    iaa.Affine(scale=(1.0, 1.5)),  # Zoom aleatório entre 1.0 e 1.5
    iaa.LinearContrast((0.5, 2.0)),  # Ajuste aleatório de contraste
    iaa.Affine(rotate=(-45, 45)),  # Rotação aleatória entre -45 e 45 graus
])

output_path = 'Diretorio\\aqui'  # pasta onde as imagens serão salvas
folder_path = 'Diretorio\\aqui'  # caminho da pasta que deseja aplicar o Data Aug

# número de imagens que irá ser gerado a partir de uma única. 
# Ex: se n_gen = 2, irá ser gerado 2 imagens artificiais para cada imagem da pasta
n_gen = 5

for img in glob(folder_path + '/*.jpg'):    
    original_image = cv2.imread(img)
    #filename = img.split('\\')[-1][:-4]  # ajuste para caminhos no Windows
    for i in range(n_gen):
        # Crie uma cópia da imagem original para aplicar a augmentação
        image = original_image.copy()
        images_aug = seq(images=[image])
        cv2.imwrite(output_path + '\\{}_{}.jpg'.format(filename, i), images_aug[0])
