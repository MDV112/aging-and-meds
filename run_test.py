import torch

with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/Jul-29-2022_11_42_19/best_ERR_diff_model.pt', 'rb') as f:
    trained_model=torch.load(f)
a=1