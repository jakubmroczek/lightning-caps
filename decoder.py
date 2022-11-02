from locale import normalize
from random import randint
from src.models.mnist_module import MNISTLitModule
from src.models.components.capsule_network import CapsuleNet

import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.nn.functional as F


from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit
from PyQt5.QtGui import QPixmap, QDoubleValidator, QIntValidator
from PyQt5.QtCore import QRect
import sys, tempfile

CONFIG_PATH = "./assets/experiment-4-default-hinton/.hydra"
CHECKPOINT_PATH = '/Users/mroczekj/Desktop/magisterka/code/lightning-hydra-template/assets/experiment-4-default-hinton/checkpoints/epoch_005.ckpt'

# CONFIG_PATH = "./assets/4-kernel-2-stride-1"
# CHECKPOINT_PATH = '/Users/mroczekj/Desktop/magisterka/code/lightning-hydra-template/assets/4-kernel-2-stride-1/epoch_001.ckpt'

def get_mnist_test_images():
    # data transformations as in learning module
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_dir = 'data'
    testset = MNIST(data_dir, train=False, transform=transform)

    return testset

def load_capsule_net_from_checkpoint():
    from hydra import compose, initialize
    
    initialize(config_path=CONFIG_PATH, job_name="capsule_net_decoder", version_base=None)
    cfg = compose(config_name="config")
    net_configuration = cfg.model.net
    
    first_capsule_layer_dimension = net_configuration.first_capsule_layer_dimension
    # first_capsule_layer_dimension = 1
    first_capusle_layer_convolution_layer_numbers = net_configuration.first_capusle_layer_convolution_layer_numbers
    output_capsules_dimension = net_configuration.output_capsules_dimension
    conv1_kernel_size = net_configuration.conv1_kernel_size
    # conv1_kernel_size = 9
    conv1_stride = net_configuration.conv1_stride
    # conv1_stride = 1
    primary_caps_kernel_size = net_configuration.primary_caps_kernel_size
    # primary_caps_kernel_size = 9
    primary_caps_stride = net_configuration.primary_caps_stride
    # primary_caps_stride = 2

    net=CapsuleNet(
        first_capsule_layer_dimension,
        first_capusle_layer_convolution_layer_numbers,
        output_capsules_dimension,
        conv1_kernel_size,
        conv1_stride,
        primary_caps_kernel_size,
        primary_caps_stride
    )

    model = MNISTLitModule.load_from_checkpoint(CHECKPOINT_PATH, net=net)

    return model

class App(QWidget):

    def __init__(self):
        super().__init__()
        # Caps net settings
        self.digit_caps_dimension = 16

        self.title = 'CAPScoder'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

        # Model
        self.model = load_capsule_net_from_checkpoint()
        self.model.eval()

        # Test dataset
        self.dataset = get_mnist_test_images()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
    
        path = 'assets/mnist/reconstuction.jpg'

        # Original image
        self.originalImage = QLabel(self)
        self.pixmap = QPixmap(path).scaled(112, 112)
        self.originalImage.setPixmap(self.pixmap)
        self.originalImage.setGeometry(150 + 112, 0, 100 + 112, 112)
        
        # Decoded image
        self.decodedImage = QLabel(self)
        self.pixmap = QPixmap(path).scaled(112, 112)
        self.decodedImage.setPixmap(self.pixmap)
        self.decodedImage.setGeometry(150 + 112 * 2, 0, 112 * 2, 112)

        # Capsule weights
        self.vbox = QVBoxLayout(self)
        self.weight_labels = [QLineEdit() for _ in range(self.digit_caps_dimension)]
        for label in self.weight_labels:
            label.setValidator(QDoubleValidator(-99.99,99.99,20))
            label.setMaximumWidth(200)
            self.vbox.addWidget(label)
            self.vbox.addStretch()

        width, height = 506, 312
        self.resize(width, height)
        
        # Load random image
        self.pushButton = QPushButton(self)
        self.pushButton.setGeometry(QRect(250, 150, 170, 28))
        self.pushButton.setText("Load random image")

        # Display with custom weigths
        self.customWeightsButton  = QPushButton(self)
        self.customWeightsButton.setGeometry(QRect(250, 200, 170, 28))
        self.customWeightsButton.setText("Render custom weights")

        # Reset to original weights
        self.resetWeights  = QPushButton(self)
        self.resetWeights.setGeometry(QRect(250, 250, 170, 28))
        self.resetWeights.setText("Reset weights")

        # Labels
        self.trueDigit = QLabel(self)
        self.trueDigit.setGeometry(QRect(250, 300, 170, 28))
        self.predicted = QLabel(self)
        self.predicted.setGeometry(QRect(250, 330, 170, 28))

        # Epsilon
        self.weight_delta = 1e-2
        self.weight_delta_text_field = QLabel(self)
        self.weight_delta_text_field.setText('Weight delta')
        self.weight_delta_text_field.setGeometry(QRect(300, 370, 170, 28))

        self.weight_delta_text_field = QLineEdit(self)
        self.weight_delta_text_field.setValidator(QDoubleValidator(-99.99,99.99,20))
        self.weight_delta_text_field.setMaximumWidth(100)
        self.weight_delta_text_field.setText(f'{self.weight_delta}')
        self.weight_delta_text_field.setGeometry(QRect(300, 400, 170, 28))

        # Modified weight index
        self.weight_index = QLabel(self)
        self.weight_index.setText('Modified weight index')
        self.weight_index.setGeometry(QRect(300, 450, 170, 28))

        self.weight_index = 0
        self.weight_index_le = QLineEdit(self)
        self.weight_index_le.setValidator(QIntValidator(0,16))
        self.weight_index_le.setMaximumWidth(100)
        self.weight_index_le.setText(f'{self.weight_index}')
        self.weight_index_le.setGeometry(QRect(300, 480, 170, 28))

        # Signals and slots
        self.pushButton.clicked.connect(self.displayImages)
        self.customWeightsButton.clicked.connect(self.displayDecodedImageFromCustomCapsuleWeights)
        self.resetWeights.clicked.connect(self.resetCapsuleWeights)

        self.show()    


    # Handle weight updates
    def wheelEvent(self, event):
        # numDegrees = event.pixelDelta() / 8
        # numSteps = numDegrees / 15

        delta = event.pixelDelta().y()
        step = float(self.weight_delta_text_field.text())
        delta *= step

        weight_index = int(self.weight_index_le.text())
        new = float(self.weight_labels[weight_index].text()) + delta
        new = str(new)
        self.weight_labels[weight_index].setText(new)

        self.displayDecodedImageFromCustomCapsuleWeights()
        # if event->orientation() == Qt.Horizontal:
        #     scrollHorizontally(numSteps)
        # else:
        #     scrollVertically(numSteps)
        event.accept()


    def displayImages(self):
        sample_idx = randint(0,10_000)
        original_image, current_y = self.dataset[sample_idx]
        
        self.current_y = current_y

        capsules, reconstructions, digit_caps = self.forward(original_image)
        self.recognized_digit = capsules[0].argmax(dim=0).item()
        
        reconstructed_image = reconstructions[0]
        self.recognized_capsule = digit_caps[0][self.recognized_digit]

        self.displayOriginalImage(original_image)
        self.displayDecodedImage(reconstructed_image)
        self.displayCapsuleWeights(self.recognized_capsule)

        self.trueDigit.setText(f'Label {self.current_y}')
        self.predicted.setText(f'Predicted {self.recognized_digit}')


    def displayOriginalImage(self, image: torch.Tensor):
        with tempfile.NamedTemporaryFile() as file:
            save_image(image, file.name, format='JPEG', normalize=True)
            pixmap = QPixmap(file.name).scaled(112, 112)
            self.originalImage.setPixmap(pixmap)

    def displayCapsuleWeights(self, capsule: torch.Tensor):
        for label, weight in zip(self.weight_labels, capsule):
            label.setText(f'{weight.item()}')

    def displayDecodedImage(self, image: torch.Tensor):
        with tempfile.NamedTemporaryFile() as file:
            save_image(image, file.name, format='JPEG', normalize=True)
            pixmap = QPixmap(file.name).scaled(112, 112)
            self.decodedImage.setPixmap(pixmap)

    def displayDecodedImageFromCustomCapsuleWeights(self):
        torch.set_printoptions(precision=10)

        new_capsule = [float(label.text()) for label in self.weight_labels]
        new_capsule = torch.FloatTensor(new_capsule)
        net_input = torch.zeros(10, self.digit_caps_dimension)
        net_input[self.recognized_digit] = new_capsule
        net_input = net_input.unsqueeze(dim=0)

        labels = torch.eye(10).index_select(dim=0, index=torch.tensor([self.recognized_digit]))
        
        y = labels

        x = net_input
        reconstructions = self.model.net.decoder((x * y[:, :, None]).reshape(x.size(0), -1))
        print(reconstructions)

        reconstructions = reconstructions.view(1, 28, 28) 

        first_image = reconstructions[0]

        with tempfile.NamedTemporaryFile() as file:
            save_image(first_image, file.name, format='JPEG', normalize=True)
            pixmap = QPixmap(file.name).scaled(112, 112)
            self.decodedImage.setPixmap(pixmap)

    def resetCapsuleWeights(self):
        self.displayCapsuleWeights(self.recognized_capsule)

    def forward(self, image):
        dummy_batch_size = 2
        # Wrap in pseudo-batch
        dummy_image  = torch.zeros(1,28,28)
        pseudo_batch = torch.cat((image, dummy_image), dim=0).unsqueeze(dim=1)
        capsules, reconstructions = self.model(pseudo_batch)

        # Directly copied from the capusle network source code
        x = F.relu(self.model.net.conv1(pseudo_batch), inplace=True)
        x = self.model.net.primary_capsules(x)
        digit_caps = self.model.net.digit_capsules(x).squeeze().transpose(0, 1)

        reconstructions = reconstructions.view(dummy_batch_size, 28, 28) 
        return capsules, reconstructions, digit_caps            


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())



