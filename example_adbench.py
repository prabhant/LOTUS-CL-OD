#git clone https://github.com/Minqi824/ADBench
#%cd ADBench
from data_generator import DataGenerator
from myutils import Utils
from lotus import LotusMetaData
from lotus import LotusModel

warnings.filterwarnings("ignore")
sys.path.append('/ADBench/')

datagenerator = DataGenerator() # data generator
utils = Utils() # utils function

# Create Metadata

data_list = ['10_cover',
 '11_donors',
 '12_fault',
 '13_fraud',
 '14_glass',
 '15_Hepatitis',
 '16_http',
 '17_InternetAds',
 '18_Ionosphere',
 '19_landsat',
 '1_ALOI',
 '20_letter',
 '21_Lymphography',
 '22_magic.gamma',
 '23_mammography']

def dataloader(dataset):
  datagenerator = DataGenerator()
  datagenerator.dataset = dataset
  data = datagenerator.generator(la=0.1, realistic_synthetic_mode=None,
                                  noise_type=None)
  return data

md = LotusMetaData(data_list[0:3], 'accuracy', 120, dataloader = dataloader, out = 'csv')
md.create_lotus_metadata()

datagenerator.dataset = '47_yeast' # specify the dataset name
data = datagenerator.generator(la=0.1, realistic_synthetic_mode=None, noise_type=None)
model = LotusModel(new_dataset=data['X_train'], meta_data_obj=md, distance = 'gwlr',
                   preprocessing = 'ica')
best_model, distance, score, dataset = model.find_model()