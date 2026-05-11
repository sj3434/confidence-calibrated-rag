"""Generate the similarity score distribution plot for the report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Real data from evaluation_summary.txt
in_domain_scores = [0.839, 0.831, 0.785, 0.870, 0.881, 0.843, 0.887, 0.779, 0.797, 0.816, 0.833, 0.759, 0.859, 0.833, 0.871, 0.862, 0.821, 0.852]
in_domain_labels = [f"Q{i}" for i in range(1, 19)]

ood_scores = [0.579, 0.597, 0.650, 0.615, 0.604, 0.590, 0.580, 0.578, 0.596, 0.598, 0.595, 0.586]
ood_labels = [f"Q{i}" for i in range(19, 31)]

tau = 0.741

fig, ax = plt.subplots(figsize=(10, 5))

# Plot individual scores as a strip/jitter plot
np.random.seed(42)
in_y = np.ones(len(in_domain_scores)) + np.random.uniform(-0.15, 0.15, len(in_domain_scores))
ood_y = np.zeros(len(ood_scores)) + np.random.uniform(-0.15, 0.15, len(ood_scores))

ax.scatter(in_domain_scores, in_y, c='#2196F3', s=80, zorder=5, edgecolors='white', linewidths=0.5, label=f'In-Domain (n=18, range [{min(in_domain_scores):.3f}, {max(in_domain_scores):.3f}])')
ax.scatter(ood_scores, ood_y, c='#F44336', s=80, zorder=5, edgecolors='white', linewidths=0.5, label=f'Out-of-Domain (n=12, range [{min(ood_scores):.3f}, {max(ood_scores):.3f}])')

# Add labels for key points
for score, label in zip(in_domain_scores, in_domain_labels):
    if label in ['Q12', 'Q10', 'Q8', 'Q7', 'Q5']:
        offset = 12 if score < 0.82 else -12
        ax.annotate(label, (score, in_y[int(label[1:])-1]), fontsize=7, ha='center', va='bottom', 
                    xytext=(0, offset), textcoords='offset points', color='#1565C0', fontweight='bold')

# Threshold line
ax.axvline(x=tau, color='#4CAF50', linewidth=2.5, linestyle='--', label=f'τ = {tau} (calibrated threshold)', zorder=4)

# Separation gap annotation
gap_left = max(ood_scores)
gap_right = min(in_domain_scores)
ax.annotate('', xy=(gap_left, 0.5), xytext=(gap_right, 0.5),
            arrowprops=dict(arrowstyle='<->', color='#FF9800', lw=2))
ax.text((gap_left + gap_right) / 2, 0.6, f'Gap = {gap_right - gap_left:.3f}', 
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#E65100')

# Shading
ax.axvspan(0.5, tau, alpha=0.08, color='#F44336', label='Abstention zone (score < τ)')
ax.axvspan(tau, 0.95, alpha=0.08, color='#2196F3', label='Generation zone (score ≥ τ)')

ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Out-of-Domain', 'In-Domain'], fontsize=11)
ax.set_xlim(0.50, 0.95)
ax.set_ylim(-0.5, 1.7)
ax.set_title('Figure 2. Similarity Score Distribution: In-Domain vs. Out-of-Domain Queries', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'score_distribution.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"✅ Plot saved to {output_path}")
