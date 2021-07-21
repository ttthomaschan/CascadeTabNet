import torch
import mmcv
import mmdet
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# Load model
config_file = '/home/elimen/Data/cascade-tabnet_pytorch/Config/cascade_mask_rcnn_hrnetv2p_w32_20e_v2.py'
checkpoint_file = '/home/elimen/Data/cascade-tabnet_pytorch/checkpoints/tablebank_both_epoch_13.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Test a single image 
img = "/home/elimen/Data/cascade-tabnet_pytorch/Demo/demo.png"
# img = "/home/elimen/Data/Project/DocTab_Infer/test_images/test_29.jpg"

# Run Inference
result = inference_detector(model, img)
print('** Results: ')
print(result)

# Visualization results
show_result_pyplot(model, img, result, score_thr=0.85)

# pt_path = "/home/elimen/Data/cascade-tabnet_pytorch/checkpoints/tablebank_both_epoch_13.pth"
# model_state_dict = torch.load(pt_path, map_location='cpu')
# # model.load_state_dict(model_state_dict)
# print(model_state_dict.keys())

# print(model_state_dict["meta"].keys())
# # print(model_state_dict["state_dict"].keys())
# print(model_state_dict["optimizer"].keys())

# state_dict_keys = model_state_dict["state_dict"].keys()
# print(type(state_dict_keys))
# with open('state_dict_tablebank.txt','w') as f:
#     for k in state_dict_keys:
#         f.write("{}\n".format(k))  
# f.close()
              