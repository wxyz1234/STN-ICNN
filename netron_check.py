import torch
from model.model import EndtoEndModel,ETE_stage1,ETE_select,ETE_stage2,label_channel,label_list,make_inverse
from model.model_ICNN import Network2
dummy_input = torch.rand(1, 3, 128, 128)

model=ETE_stage1(torch.device("cpu"))
onnx_path = "onnx_ETE_stage1.onnx"
#model=Network2();
#onnx_path = "onnx_ICNN_stage1.onnx"
torch.onnx.export(model, dummy_input, onnx_path)
