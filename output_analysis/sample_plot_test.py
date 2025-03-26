import matplotlib.pyplot as plt
import torch


metrics = {
    'classes': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32),
    'map': torch.tensor(0.3156),
    'map_50': torch.tensor(0.4107),
    'map_75': torch.tensor(0.2414),
    'map_large': torch.tensor(0.5598),
    'map_medium': torch.tensor(0.3767),
    'map_per_class': torch.tensor([0.3970, 0.3424, 0.3766, 0.3588, 0.2619, 0.3874, 0.2344, 0.5421, 0.0000, 0.2559]),
    'map_small': torch.tensor(0.1379),
    'mar_1': torch.tensor(0.1879),
    'mar_10': torch.tensor(0.3399),
    'mar_100': torch.tensor(0.3523),
    'mar_100_per_class': torch.tensor([0.4413, 0.3874, 0.4237, 0.3942, 0.3043, 0.4408, 0.2726, 0.5690, 0.0000, 0.2900]),
    'mar_large': torch.tensor(0.5993),
    'mar_medium': torch.tensor(0.4243),
    'mar_small': torch.tensor(0.1607)
    }

# Convert tensors to lists or scalars for plotting
classes = metrics['classes'].tolist()
map_per_class = metrics['map_per_class'].tolist()
mar_100_per_class = metrics['mar_100_per_class'].tolist()

# Plot aggregated metrics (map and mar)
def plot_aggregated_metrics(metrics):
    labels = ['mAP', 'mAP_50', 'mAP_75', 'mAP_Large', 'mAP_Medium', 'mAP_Small',
              'mAR_1', 'mAR_10', 'mAR_100', 'mAR_Large', 'mAR_Medium', 'mAR_Small']
    values = [metrics['map'].item(), metrics['map_50'].item(), metrics['map_75'].item(),
              metrics['map_large'].item(), metrics['map_medium'].item(), metrics['map_small'].item(),
              metrics['mar_1'].item(), metrics['mar_10'].item(), metrics['mar_100'].item(),
              metrics['mar_large'].item(), metrics['mar_medium'].item(), metrics['mar_small'].item()]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Metric Values')
    plt.title('Aggregated Metrics (mAP and mAR)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plot per-class metrics
def plot_per_class_metrics(classes, map_per_class, mar_100_per_class):
    plt.figure(figsize=(12, 6))
    x = range(len(classes))

    plt.bar(x, map_per_class, width=0.4, label='mAP per class', color='orange')
    plt.bar([i + 0.4 for i in x], mar_100_per_class, width=0.4, label='mAR_100 per class', color='purple')

    plt.xticks([i + 0.2 for i in x], classes, rotation=45)
    plt.xlabel('Classes')
    plt.ylabel('Metric Values')
    plt.title('Per-Class Metrics (mAP and mAR_100)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plot metrics
plot_aggregated_metrics(metrics)
plot_per_class_metrics(classes, map_per_class, mar_100_per_class)