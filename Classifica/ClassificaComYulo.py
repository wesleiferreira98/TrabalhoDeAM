import cv2
import numpy as np
import os
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GdkPixbuf, Gdk

class ImageProcessorApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Detecção de Objetos")
        self.set_default_size(800, 600)

        self.image_path = None
        self.output_dir = None

        self.init_ui()

    def init_ui(self):
        self.grid = Gtk.Grid()
        self.grid.set_column_homogeneous(True)
        self.grid.set_row_spacing(10)

        self.select_images_button = Gtk.Button(label="Selecionar Imagens")
        self.select_images_button.connect("clicked", self.on_open_image)
        self.grid.attach(self.select_images_button, 0, 0, 1, 1)

        self.select_output_dir_button = Gtk.Button(label="Selecionar Diretório de Saída")
        self.select_output_dir_button.connect("clicked", self.on_select_output_dir)
        self.grid.attach(self.select_output_dir_button, 1, 0, 1, 1)

        self.process_button = Gtk.Button(label="Processar Imagens")
        self.process_button.connect("clicked", self.on_detect_objects)
        self.grid.attach(self.process_button, 0, 1, 2, 1)

        self.progress_bar = Gtk.ProgressBar()
        self.grid.attach(self.progress_bar, 0, 2, 2, 1)

        self.image_view = Gtk.Image()
        self.grid.attach(self.image_view, 0, 3, 2, 1)

        self.set_border_width(10)
        self.add(self.grid)

    def on_open_image(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Selecione uma imagem",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
            buttons=(
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_OPEN, Gtk.ResponseType.OK
            )
        )
        dialog.set_default_response(Gtk.ResponseType.OK)

        filter_img = Gtk.FileFilter()
        filter_img.set_name("Imagens")
        filter_img.add_mime_type("image/*")
        dialog.add_filter(filter_img)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.image_path = dialog.get_filename()
            self.show_image()
        dialog.destroy()

    def on_select_output_dir(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Selecione o Diretório de Saída",
            parent=self,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
            buttons=(
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_OPEN, Gtk.ResponseType.OK
            )
        )
        dialog.set_default_response(Gtk.ResponseType.OK)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.output_dir = dialog.get_filename()
        dialog.destroy()

    def on_detect_objects(self, widget):
        if self.image_path and self.output_dir:
            image = cv2.imread(self.image_path)
            self.detect_objects(image)

    def detect_objects(self, image):
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        processed_image_path = os.path.join(str(self.output_dir), "processed_image.jpg")
        cv2.imwrite(processed_image_path, image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = GdkPixbuf.Pixbuf.new_from_data(
            image.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, width, height, width * 3
        )

        self.image_view.set_from_pixbuf(image)
        self.show_all()

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.connect("destroy", Gtk.main_quit)
    Gtk.main()
