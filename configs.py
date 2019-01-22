PATH_TO_IMAGES_DIR = './dataset_images'
PATH_TO_TRAIN_FILE = './dataset_text/train.txt'
PATH_TO_VAL_FILE = './dataset_text/val.txt'
PATH_TO_TEST_FILE = './dataset_text/test.txt'

PRE_TRAINED_WEIGHTS_FOR_HEAT_MAP = './model0122.pth.tar'



CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

LIST_NN_ARCHITECTURE = ['DENSE-NET-121', 'DENSE-NET-169', 'DENSE-NET-201']
NN_ARCHITECTURE = LIST_NN_ARCHITECTURE[0]
NN_IS_PRE_TRAINED = True

NUM_CLASSES = 14
TRANS_RESIZE = 256
TRANS_CROP = 224

TRAIN_DICT = {
    'Batch Size': 32,
    'Max Epoch': 1000,
    'Learning Rate': 0.001
}

TEST_DICT = {
    'Batch Size': 16,
}

