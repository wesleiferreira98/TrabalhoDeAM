import os
import cv2
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
#Gustavo
class ImageProcessorApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Processador de Imagens")
        self.set_default_size(400, 200)
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

            # Converta a imagem para escala de cinza
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Aplique a detecção de bordas (Canny edge detection)
            edges = cv2.Canny(gray_image, threshold1=30, threshold2=150)

            output_path = os.path.join(self.output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, edges)

win = ImageProcessorApp()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
