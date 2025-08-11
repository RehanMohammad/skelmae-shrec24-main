import matplotlib.pyplot as plt
import seaborn as sns

def plot_temporal_heatmap(attn_weights, frame_idx=0, head_idx=0):
    """
    attn_weights: [B, V, heads, V] from AdaptiveTemporal
    """
    plt.figure(figsize=(10, 8))
    data = attn_weights[frame_idx, :, head_idx].detach().cpu().numpy()
    
    sns.heatmap(data, cmap="viridis", 
                xticklabels=range(data.shape[1]),
                yticklabels=range(data.shape[0]))
    
    plt.title(f"Temporal Attention @ Frame {frame_idx} (Head {head_idx})")
    plt.xlabel("Target Joints (t+1)")
    plt.ylabel("Source Joints (t)")
    plt.savefig(f"attention_f{frame_idx}_h{head_idx}.png")
    
def plot_joint_importance(model):
    if hasattr(model, 'joint_adapter'):
        imp = model.joint_adapter.importance.detach().cpu().numpy()
        plt.bar(range(len(imp)), imp)
        plt.title("Learned Joint Importance")
        plt.xlabel("Joint Index")
        plt.ylabel("Importance Weight")
        plt.savefig("joint_importance.png")