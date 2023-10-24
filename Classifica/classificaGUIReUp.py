import os
import cv2
import gi
import numpy as np
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

class ImageProcessorApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Processador de Imagens")
        self.set_default_size(400, 200)
        self.gray_img=None
        self.selected_images = []
        self.output_dir = ""

        self.init_ui()

    def init_ui(self):
        grid = Gtk.Grid()
        grid.set_column_homogeneous(True)

        self.select_images_button = Gtk.Button(label="Selecionar Imagens")
        self.select_images_button.connect("clicked", self.select_images)
        grid.attach(self.select_images_button, 0, 0, 1, 1)

        self.select_output_dir_button = Gtk.Button(label="Selecionar Diretório de Saída")
        self.select_output_dir_button.connect("clicked", self.select_output_dir)
        grid.attach(self.select_output_dir_button, 1, 0, 1, 1)

        self.process_button = Gtk.Button(label="Processar Imagens")
        self.process_button.connect("clicked", self.process_images)
        grid.attach(self.process_button, 0, 1, 2, 1)

        self.add(grid)

    def select_images(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Selecionar Imagens",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
            buttons=(
                "Cancelar", Gtk.ResponseType.CANCEL,
                "Selecionar", Gtk.ResponseType.OK
            )
        )
        dialog.set_select_multiple(True)
        dialog.set_default_response(Gtk.ResponseType.OK)
        filter_images = Gtk.FileFilter()
        filter_images.set_name("Images")
        filter_images.add_mime_type("image/jpeg")
        filter_images.add_mime_type("image/png")
        filter_images.add_mime_type("image/bmp")
        filter_images.add_mime_type("image/jpg")
        dialog.add_filter(filter_images)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.selected_images = dialog.get_filenames()
        dialog.destroy()

    def select_output_dir(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Selecionar Diretório de Saída",
            parent=self,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
            buttons=(
                "Cancelar", Gtk.ResponseType.CANCEL,
                "Selecionar", Gtk.ResponseType.OK
            )
        )
        dialog.set_default_response(Gtk.ResponseType.OK)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.output_dir = dialog.get_filename()
        dialog.destroy()

    def process_images(self, widget):
        if not self.selected_images or not self.output_dir:
            return

        target_size = (3024, 4032)  # Tamanho desejado das imagens
        for image_path in self.selected_images:
            img = cv2.imread(image_path)
            img = cv2.resize(img, target_size)

            # Aplicar outras etapas de processamento aqui
            img = self.apply_color_conversion(img)
            img = self.apply_noise_removal(img)
            img = self.apply_histogram_equalization(img)
            img = self.add_border_around_leaves(img)

            output_path = os.path.join(self.output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, img)

    def apply_color_conversion(self, img):
        # Exemplo: Converta a imagem para escala de cinza (grayscale)
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.gray_img

    def apply_noise_removal(self, img):
        # Exemplo: Aplique um filtro de suavização (blur) para remover ruído
        smoothed_img = cv2.GaussianBlur(img, (5, 5), 0)
        return smoothed_img

    def apply_contrast_adjustment(self, img):
        # Exemplo: Ajuste o contraste e o brilho da imagem
        alpha = 1.5  # Ajuste o valor conforme necessário
        beta = 10  # Ajuste o valor conforme necessário
        adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return adjusted_img

    def apply_histogram_equalization(self, img):
        # Verifique se a imagem está em escala de cinza
        if img.shape[-1] == 1:
            # A imagem já está em escala de cinza, não é necessário converter
            equalized_img = cv2.equalizeHist(img)
        else:
            # Se a imagem não estiver em escala de cinza, converta-a
            equalized_img = cv2.equalizeHist(self.gray_img)
        return equalized_img

    def apply_edge_filtering(self, img):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        return sobel_combined

    def apply_intensity_normalization(self, img):
        # Exemplo: Normalize a intensidade dos pixels para o intervalo [0, 1]
        normalized_img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        return normalized_img

    def apply_segmentation(self, img):
        # Aplique a segmentação colorida, você pode usar métodos como a detecção de cor ou outras técnicas para segmentar as áreas de interesse.
        segmented_img = self.color_segmentation(img)
        return segmented_img

    def color_segmentation(self, img):
        # Certifique-se de que a imagem está no formato CV_8U
        img = cv2.convertScaleAbs(img)

        # Converta a imagem BGR para o espaço de cores HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Ajuste os valores de detecção de verde para tons mais claros
        # Ajuste os valores conforme necessário
        lower_green = np.array([30, 50, 50], dtype=np.uint8)
        upper_green = np.array([100, 255, 255], dtype=np.uint8)


        # Crie uma máscara que filtra as cores verdes
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Aplique a máscara à imagem original
        segmented_img = cv2.bitwise_and(img, img, mask=mask)

        return segmented_img

    def add_border_around_leaves(self, img):
        smoothed_img = cv2.GaussianBlur(self.gray_img, (15, 15), 0)

        # Realize a segmentação das folhas (você pode ajustar o limite conforme necessário)
        _, binary_image = cv2.threshold(smoothed_img, 148, 255, cv2.THRESH_BINARY)

        # Encontre os contornos das folhas
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhe os contornos das folhas e adicione uma borda
        img_with_contours = img.copy()
        border_color = (0, 0, 0)  # Cor da borda (preta)
        border_thickness = 6  # Espessura da borda
        cv2.drawContours(img_with_contours, contours, -1, border_color, border_thickness)

        return img_with_contours


    def get_pixel(self, event, x, y, flags, param, img):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = img[y, x]
            print("Valores de Pixel (BGR) em ({}, {}): {}".format(x, y, pixel))

    def remove_artifacts(self, img):
        lower_reflection = np.array([200, 200, 200], dtype=np.uint8)
        upper_reflection = np.array([255, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(img, lower_reflection, upper_reflection)
        mask_outside_range = cv2.bitwise_not(mask)
        img[mask_outside_range > 0] = [0, 0, 0]

win = ImageProcessorApp()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
