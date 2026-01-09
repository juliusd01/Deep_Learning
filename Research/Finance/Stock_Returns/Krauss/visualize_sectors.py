import torch
import matplotlib.pyplot as plt
import numpy as np


model = torch.load("Research/Finance/Stock_Returns/Krauss/models/lstm/sector/h_25_l_1_lr_0.001_lag_60/best_lstm_model_2020.pth")
sector_weights = model["sector_embedding.weight"]

# Sector mapping for 10 sectors: 
sector_mapping_10 = {0: 'Basic Materials', 1: 'Consumer Cyclicals', 2: 'Consumer Non-Cyclicals', 3: 'Energy', 
4: 'Financials', 5: 'Healthcare', 6: 'Industrials', 7: 'Real Estate', 8: 'Technology', 9: 'Utilities'}

# Sector mapping for 11 sectors:
sector_mapping_11 = {0: "Academic & Educational Services", 1: "Basic Materials", 2: "Consumer Cyclicals", 3: "Consumer Non-Cyclicals",
4: "Energy", 5: "Financials", 6: "Healthcare", 7: "Industrials", 8: "Real Estate", 9: "Technology", 10: "Utilities"}

# Convert to numpy
weights_np = sector_weights.cpu().numpy()

# Determine which mapping to use based on the number of sectors
num_sectors = weights_np.shape[0]
sector_mapping = sector_mapping_11 if num_sectors == 11 else sector_mapping_10
sector_names = [sector_mapping[i] for i in range(num_sectors)]

# Create 3 plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Heatmap of all weights
im = axes[0].imshow(weights_np, cmap='RdBu_r', aspect='auto')
axes[0].set_yticks(range(len(sector_names)))
axes[0].set_yticklabels(sector_names)
axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(['Dim 1', 'Dim 2', 'Dim 3'])
axes[0].set_title('Sector Embedding Weights Heatmap')
plt.colorbar(im, ax=axes[0])

# Plot 2: Bar plot for each dimension
x = np.arange(len(sector_names))
width = 0.25
axes[1].bar(x - width, weights_np[:, 0], width, label='Dim 1', alpha=0.8)
axes[1].bar(x, weights_np[:, 1], width, label='Dim 2', alpha=0.8)
axes[1].bar(x + width, weights_np[:, 2], width, label='Dim 3', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(sector_names, rotation=45, ha='right')
axes[1].set_ylabel('Weight Value')
axes[1].set_title('Sector Weights by Dimension')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Plot 3: Individual lines for each dimension across sectors
for i in range(3):
    axes[2].plot(range(len(sector_names)), weights_np[:, i], marker='o', label=f'Dim {i+1}', linewidth=2)
axes[2].set_xticks(range(len(sector_names)))
axes[2].set_xticklabels(sector_names, rotation=45, ha='right')
axes[2].set_ylabel('Weight Value')
axes[2].set_title('Sector Weights Across Dimensions')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('Research/Finance/Stock_Returns/Krauss/img/sector_weights_visualization.png', dpi=300, bbox_inches='tight')
plt.show()



# Create 3D plot of sector vectors
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each sector as a point and vector from origin
colors = plt.cm.tab10(np.linspace(0, 1, len(sector_names)))

for i, (sector, color) in enumerate(zip(sector_names, colors)):
    x, y, z = weights_np[i]
    # Plot vector from origin
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.1, linewidth=2, alpha=0.8)
    # Plot point at the end
    ax.scatter(x, y, z, color=color, s=100, alpha=0.8, edgecolors='black', linewidth=1)
    ax.text(x, y, z, f'  {sector}', fontsize=9, ha='left')

# Add origin point
ax.scatter(0, 0, 0, color='black', s=100, marker='o', label='Origin')
ax.set_xlabel('Dimension 1', fontsize=11)
ax.set_ylabel('Dimension 2', fontsize=11)
ax.set_zlabel('Dimension 3', fontsize=11)
ax.set_title('3D Visualization of Sector Embedding Vectors', fontsize=14, pad=20)
ax.grid(True, alpha=0.3)
# Set equal aspect ratio for better visualization
max_range = np.abs(weights_np).max()
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

plt.savefig('Research/Finance/Stock_Returns/Krauss/img/sector_weights_3d.png', dpi=300, bbox_inches='tight')
plt.show()
