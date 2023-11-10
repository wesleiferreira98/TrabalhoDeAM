import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib


class ImageProcessorApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Treino do modelo")
        self.set_default_size(520, 320)
        self.set_resizable(False)
        self.selected_images = []
        self.output_dir = ""
        self.model = None  # Adicionado para armazenar o modelo treinado
        self.accuracy_label = Gtk.Label(label="Acurácia do modelo: ")
        self.f1_score_label = Gtk.Label(label="F1-score do modelo: ")

        self.init_ui()

    def init_ui(self):
        self.grid = Gtk.Grid()
        self.grid.set_column_homogeneous(True)
        self.grid.set_row_spacing(10)  # Ajusta o espaçamento vertical entre as linhas

        self.select_images_button = Gtk.Button(label="Selecionar Arquivos .txt")
        self.select_images_button.connect("clicked", self.select_txt_files)
        self.grid.attach(self.select_images_button, 0, 0, 1, 1)

        self.train_model_button = Gtk.Button(label="Treinar Modelo")
        self.train_model_button.connect("clicked", self.train_model)
        self.grid.attach(self.train_model_button, 1, 0, 1, 1)

        self.progress_bar = Gtk.ProgressBar()
        self.grid.attach(self.progress_bar, 0, 2, 2, 1)

        self.accuracy_label = Gtk.Label(label="Acurácia do modelo: ")
        self.grid.attach(self.accuracy_label, 0, 3, 2, 1)

        self.f1_score_label = Gtk.Label(label="F1-score do modelo: ")
        self.grid.attach(self.f1_score_label, 0, 4, 2, 1)

        self.add(self.grid)
        
        # Ajusta o espaçamento horizontal entre as colunas
        self.grid.set_column_spacing(10)

        # Ajusta o espaçamento externo (borda) da janela
        self.set_border_width(10)

        # Define o tamanho da janela
        self.set_size_request(520, 320)


    def select_txt_files(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Selecionar Arquivos .txt",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
            buttons=(
                "Cancelar", Gtk.ResponseType.CANCEL,
                "Selecionar", Gtk.ResponseType.OK
            )
        )
        dialog.set_select_multiple(True)
        dialog.set_default_response(Gtk.ResponseType.OK)
        filter_txt = Gtk.FileFilter()
        filter_txt.set_name("Arquivos .txt")
        filter_txt.add_mime_type("text/plain")
        filter_txt.add_pattern("*.txt")
        dialog.add_filter(filter_txt)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.selected_images = dialog.get_filenames()
        dialog.destroy()

    def select_output_dir(self, widget):
        # Lógica para selecionar o diretório de saída
        pass

    def train_model(self, widget):
        # Verifique se há pelo menos um arquivo selecionado
        if not self.selected_images:
            return

        # Inicialize as listas para armazenar rótulos e características de todos os arquivos
        all_labels = []
        all_features = []

        # Carregue dados de cada arquivo selecionado
        for txt_file in self.selected_images:
            data = np.loadtxt(txt_file)
            print("O arquivo atual é: ",txt_file)

            # Separar rótulos e características
            labels = data[:, 0]  # Primeira coluna é o rótulo
            features = data[:, 1:]  # Restantes colunas são características

            # Adicione rótulos e características à lista global
            all_labels.extend(labels)
            all_features.extend(features)

        # Convertemos as listas em arrays NumPy
        all_labels = np.array(all_labels)
        all_features = np.array(all_features)

        # Divisão em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, all_labels, test_size=0.2, random_state=42
        )

         # Crie e treine o modelo
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_test_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        f1 = f1_score(y_test, self.model.predict(X_test), average='weighted')

        precision_per_class = precision_score(y_test, y_test_pred, average=None)
        recall_per_class = recall_score(y_test, y_test_pred, average=None)

        # Arredonde os valores para 2 casas decimais
        rounded_accuracy = round(accuracy, 2)
        rounded_f1 = round(f1, 2)

        # Arredonde a precisão e revocação para 2 casas decimais e trate o valor indefinido como 0
        rounded_precision = [round(p, 2) if not np.isnan(p) else 0.0 for p in precision_per_class]
        rounded_recall = [round(r, 2) if not np.isnan(r) else 0.0 for r in recall_per_class]

        # Atualize as labels de acurácia, precisão, revocação e F1-score
        self.accuracy_label.set_label(f'Acurácia do modelo: {rounded_accuracy}')
        self.f1_score_label.set_label(f'F1-score do modelo: {rounded_f1}')

        # Crie um gráfico de barras com cores diferentes
        fig, ax = plt.subplots()
        metrics = ['Acurácia', 'F1-score']
        values = [rounded_accuracy, rounded_f1]

        # Adicione precisão e revocação ao gráfico
        for i, (prec, rec) in enumerate(zip(rounded_precision, rounded_recall)):
            label = f'Classe {i + 1}'
            
            # Verifique se o valor é diferente de 0 antes de adicionar ao gráfico
            if prec != 0.0:
                metrics.append('Precisão')
                values.append(prec)

            if rec != 0.0:
                metrics.append('Revocação')
                values.append(rec)

        colors = ['blue', 'green', 'orange', 'red'][:len(metrics)]
        bars = ax.bar(metrics, values, color=colors)

        # Adicione rótulos às barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{value}', ha='center', va='bottom')

# ...


        # Adicione o gráfico à interface
        chart_canvas = FigureCanvas(fig)
        chart_widget = Gtk.ScrolledWindow()
        chart_widget.set_size_request(300, 200)
        chart_widget.add(chart_canvas)
        
        # Adicione o widget do gráfico diretamente ao grid
        self.grid.attach(chart_widget, 0, 6, 2, 1)
        self.grid.show_all()  # Atualiza a interface para mostrar o gráfico


    
    





if __name__ == "__main__":
    app = ImageProcessorApp()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()
