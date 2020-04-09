import torch.nn as nn
import torch.quantization
from loader import get_imagenet_1k_loaders
from models.mobilenetv2 import load_model, print_size_of_model
from examples.helper import evaluate, run_benchmark

# # Setup warnings
import warnings

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)

device = 'cpu'
data_path = '../datasets/imagenet_1k/'
saved_model_dir = '../pretrain/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 30

data_loader, data_loader_test = get_imagenet_1k_loaders(data_path,
                                                        train_batch_size,
                                                        eval_batch_size)
criterion = nn.CrossEntropyLoss()

# MobileNetv2 Model Download
# https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
float_model = load_model(saved_model_dir + float_model_file).to(device)

print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# Fuses modules
float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
print('\n Inverted Residual Block: After fusion\n\n', float_model.features[1].conv)

print("Size of baseline model")
print_size_of_model(float_model)
num_eval_batches = 10
top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)
