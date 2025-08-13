import os
import numpy as np
import torch
import tqdm
import os.path as opt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.temporal_teacher import velocity_similarity_teacher


LABELS= ["no_gesture",
                            "point_1f",
                            "point_2f",
                            "click_1f",
                            "click_2f",
                            "throw_up",
                            "throw_down",
                            "throw_left",
                            "throw_right",
                            "open_twice",
                            "double_click_1f",
                            "double_click_2f",
                            "zoom_in",
                            "zoom_out",
                            ]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), #thumb
    (0,5),(5,6),(6,7),(7,8), #index
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20) #pinky
]


_TEMPORAL_RULES = {
     0: [ 0,  1,  5,  9, 13, 17,  4,  8, 12, 16, 20],
     1: [ 0,  1,  2,  5],
     2: [ 1,  2,  3],
     3: [ 2,  3,  4,  0],
     4: [ 0,  3,  4],
     5: [ 0,  1,  5, 9,  6],
     6: [ 5,  6, 7],
     7: [ 6,  7, 8,  0],
     8: [ 0,  7, 8],
     9: [ 0,  5, 9, 13, 10],
    10: [ 9, 10, 11],
    11: [10, 11, 12,  0],
    12: [11,  12, 0],
    13: [ 0,  9, 13, 17, 14],
    14: [13, 14, 15],
    15: [14, 16,  15, 0],
    16: [ 0, 15, 16],
    17: [ 0, 13, 17, 18],
    18: [19, 17, 18],
    19: [20, 18, 19],
    20: [ 0, 19, 20],
}


def top_k(array, k):
        flat = array.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        indices = indices[np.argsort(-flat[indices])]
        return np.sort(np.unravel_index(indices, array.shape))
    
def interpolate_landmarks(landmarks, L):
        l, n_landmarks, _ = landmarks.shape
        assert l > 1, "The sequence of landmarks should have at least two landmarks"

        # Compute the indices of the input landmarks
        input_indices = np.linspace(0, l - 1, l, dtype=int)

        # Compute the indices of the output landmarks
        output_indices = np.linspace(0, l - 1, L, dtype=float)

        # Compute the fractional part of the output indices
        fractions = output_indices % 1

        # Compute the integer part of the output indices
        output_indices = np.floor(output_indices).astype(int)

        # Initialize the output array
        interpolated_landmarks = np.zeros((L, n_landmarks,4), dtype=float)

        # Compute the interpolated landmarks
        for i in range(L):
            if fractions[i] == 0:
                # The output index corresponds to an input landmark, so just copy it
                interpolated_landmarks[i] = landmarks[input_indices[output_indices[i]]]
            else:
                # Compute the mean vector between the two nearest input landmarks
                v1 = landmarks[input_indices[output_indices[i]]]
                v2 = landmarks[input_indices[min(output_indices[i] + 1, l - 1)]]
                interpolated_landmarks[i] = np.mean([v1, v2], axis=0)

        return interpolated_landmarks

# Sanity‑check: every joint is present
assert set(_TEMPORAL_RULES.keys()) == set(range(21)), "Rules must cover 0‑20 joints"


# ------------------------------------------------------------------
# 2. Graph builder
# ------------------------------------------------------------------
def build_temporal_edges(
    T: int,
    device="cpu",
    model=None,
    x: torch.Tensor=None,
    threshold: float = 0.5,
    return_scores: bool = False,     # NEW
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Build temporal adjacency A[t] for links from frame t -> t+1.

    Returns:
        - if return_scores=False (default): Bool tensor [T, V, V]
        - if return_scores=True: (A_bool [T,V,V], scores_full [T,V,V])
          where scores_full[T-1] == 0 (padded last frame)
    """
    V = 21
    A = torch.zeros((T, V, V), dtype=torch.bool, device=device)

    # Fixed rules fallback
    if model is None or x is None:
        for t in range(T - 1):
            for src, dest_list in _TEMPORAL_RULES.items():
                for dst in dest_list:
                    A[t, src, dst] = True
        if return_scores:
            scores_full = A.float()   # fixed rules: treat as 0/1 scores
            return A, scores_full
        return A

    # Ensure [B, C, T, V]
    if x.dim() == 3:  # [C,T,V]
        x = x.unsqueeze(0)
    assert x.dim() == 4
    B, C, T_in, V_in = x.shape
    assert T_in == T and V_in == V

    xyz = x[:, :3, :, :]  # [B,3,T,V]

    model_was_training = getattr(model, "training", False)
    if hasattr(model, "eval"): model.eval()
    attn_ts = []
    with torch.no_grad():
        for t in range(T - 1):
            cur = xyz[:, :, t,  :].permute(0, 2, 1)  # [B,V,3]
            nxt = xyz[:, :, t+1,:].permute(0, 2, 1)  # [B,V,3]
            attn_t = model(cur, nxt)                 # [B, V, heads, V] or [B,V,V]
            if attn_t.dim() == 4:
                attn_t = attn_t.mean(dim=2)         # avg heads -> [B,V,V]
            attn_ts.append(attn_t)
    if model_was_training and hasattr(model, "train"): model.train()

    # [B, T-1, V, V] -> [T-1, V, V] by batch mean
    scores = torch.stack(attn_ts, dim=1).mean(dim=0)           # [T-1, V, V]
    A[:-1] = scores > threshold

    if return_scores:
        scores_full = torch.zeros((T, V, V), device=device, dtype=scores.dtype)
        scores_full[:-1] = scores
        return A, scores_full
    return A



def get_adjacency_matrix(connections):
    n_lands = len(set([joint for tup in connections for joint in tup]))
    adj = np.zeros((n_lands, n_lands))
    for i, j in connections:
        adj[i][j] = 1.
        adj[j][i] = 1.
    adj += np.eye(adj.shape[0])
    return adj

def calculate_connectivity(adj_matrix, edges):
    connectivity = {i: 0 for i in range(len(adj_matrix))}
    for edge in edges:
        connectivity[edge[0]] += 1
    return connectivity


IPN_AM = get_adjacency_matrix(HAND_CONNECTIONS)
IPN_AM = torch.from_numpy(IPN_AM).float()
CONNECTIVITY = calculate_connectivity(IPN_AM, HAND_CONNECTIONS) 

class IPNDataset(Dataset):
    """
    Reads IPN annotations (.csv) and landmark .txt files.
    Returns: {'Sequence': Tensor(C=5, T=max_seq, V=21), 'Label': LongTensor}
    """
    def __init__(self, data_dir, ann_file, max_seq=4, normalize=True,
                 temporal_edge_model=None, edge_threshold=0.5):
        self.data_dir  = data_dir
        self.ann_file  = ann_file
        self.max_seq   = max_seq
        self.normalize = normalize
        self.label_map = LABELS
        self.connectivity = {i:0 for i in range(21)}
        self.temporal_edge_model = temporal_edge_model
        self.edge_threshold = edge_threshold
        self.samples = []

        assert opt.exists(data_dir), 'path {} does not exist'.format(data_dir)   
             
        # precompute connectivity count per joint
        
        for p,c in HAND_CONNECTIONS:
            self.connectivity[p] += 1
            self.connectivity[c] += 1


        # load all (feature, label) samples
        
        with open(self.ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                folder = parts[0]
                seq_id, gesture, s, e, extra = parts[1:6]
                # label = int(gesture) - 1
                fname = f"{seq_id}_{gesture}_{s}_{e}_{extra}.txt"
                path  = os.path.join(self.data_dir, folder, fname)
                feats = self._load_file(path)
                if feats is not None:
                    # self.samples.append((feats, label))
                    self.samples.append(feats)

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {ann_file}")

    def _load_file(self, path):
        if not os.path.exists(path):
            return None

        # split into raw frame blocks
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_blocks = [blk for blk in f.read().split('\n\n') if blk.strip()]
            raw_blocks = raw_blocks [:-1]

        frames = []
        for blk in raw_blocks:
            lines = [l for l in blk.split('\n')if l.strip()]
            pts = []
            for i in lines :
                if len(i)==1:
                    coords = [-1.0, -1.0, -1.0, -1.0]
                else:
                    coords = i.split(';')
                    coords = list(filter(lambda x: len(x), coords))
                    coords = [float(x) for x in coords] 
                pts.append(coords[:3])
            arr = np.asarray(pts, dtype=np.float32)
            if arr.shape == (21, 3):
                frames.append(arr)

        if len(frames) < 2:
            return None  # not enough frames

        seq = np.stack(frames, axis=0)  
        return self._post_process(seq)

    
    def _post_process(self, seq):
        
        feats = seq.copy()

        # normalize XYZ if desired
        if self.normalize:
            m, s = feats.mean(), feats.std()
            feats = (feats - m) / (s + 1e-6)

        T = feats.shape[0]
        # top‐k by movement magnitude if too long
        if T > self.max_seq:
            delta = np.linalg.norm(seq[1:] - seq[:-1], axis=(1,2))
            idxs  = np.argsort(-delta)[: self.max_seq]
            idxs  = np.sort(np.concatenate(([0], idxs)))[: self.max_seq]
            feats = feats[idxs]
            T = self.max_seq

        # interpolate if too short
        if T != self.max_seq:
            old = np.linspace(0, 1, T)
            new = np.linspace(0, 1, self.max_seq)
            tmp = np.zeros((self.max_seq, 21, 3), dtype=feats.dtype)
            for v in range(21):
                for c in range(3):
                    tmp[:, v, c] = np.interp(new, old, feats[:, v, c])
            feats = tmp
        
        return feats  # (max_seq, 21, 3)

        # return feats


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats = self.samples[idx]
        x = torch.from_numpy(feats).permute(2, 0, 1)  # [3, T, V]
        teacher = velocity_similarity_teacher(x.float(), tau=0.5, topk=4, remove_self=True)  # [T, V, V]


        A_temporal, A_temporal_scores = build_temporal_edges(
            self.max_seq,
            device=x.device,
            model=self.temporal_edge_model,
            x=x,
            threshold=self.edge_threshold,
            return_scores=True,                         # NEW
        )
        A_spatial = IPN_AM.to(x.device)

        return {
            'Sequence': x,                   # [3, T, V]
            'A_spatial': A_spatial,          # [V, V]
            'A_temporal_teacher': teacher,   # [T, V, V] (soft probs), last frame zero
            'A_temporal': A_temporal, 'A_temporal_scores': A_temporal_scores, 
        }

    

    ####################################

class FinetuningDataset(Dataset):
    
    def __init__(self, data_dir, ann_file, max_seq=80, normalize=False,
                 temporal_edge_model=None, edge_threshold=0.5):
        assert opt.exists(data_dir), f'path {data_dir} does not exist'
        self.data_dir  = data_dir
        self.ann_file  = ann_file
        self.normalize = normalize
        self.max_seq   = max_seq
        self.label_map = LABELS
        self.temporal_edge_model = temporal_edge_model
        self.edge_threshold = edge_threshold
        self.samples = []
        with open(self.ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                folder = parts[0]
                seq_id, gesture, s, e, extra = parts[1:6]
                label = int(gesture) - 1
                fname = f"{seq_id}_{gesture}_{s}_{e}_{extra}.txt"
                path  = os.path.join(self.data_dir, folder, fname)
                feats = self._load_file(path)
                if feats is not None:
                    self.samples.append((feats, label))

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.ann_file}")
    
    def _load_file(self, path):
        if not os.path.exists(path):
            return None

        # split into raw frame blocks
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_blocks = [blk for blk in f.read().split('\n\n') ]#if blk.strip()]
            raw_blocks = raw_blocks [:-1]


        frames = []
        for blk in raw_blocks:
            lines = [l for l in blk.split('\n')]#if l.strip()]
            pts = []
            for i in lines :
                if len(i)==1:
                    coords = [-1.0, -1.0, -1.0, -1.0]
                else:
                    coords = i.split(';')
                    coords = list(filter(lambda x: len(x), coords))
                    coords = [float(x) for x in coords] 
                pts.append(coords)
            arr = np.asarray(pts, dtype=np.float32)
            if arr.shape == (21, 3):
                frames.append(arr)

        if len(frames) < 2:
            return None  # not enough frames

        seq = np.stack(frames, axis=0)  # (T,21,5)
        return self._post_process(seq)

    def _post_process(self, seq):

        feats = seq.copy()
        # normalize XYZ if desired
        if self.normalize:
            m, s = feats.mean(), feats.std()
            feats = (feats - m) / (s + 1e-6)

        T = feats.shape[0]
        # top‐k by movement magnitude if too long
        if T > self.max_seq:
            delta = np.linalg.norm(seq[1:] - seq[:-1], axis=(1,2))
            idxs  = np.argsort(-delta)[: self.max_seq]
            idxs  = np.sort(np.concatenate(([0], idxs)))[: self.max_seq]
            feats = feats[idxs]
            T = self.max_seq

        # interpolate if too short
        if T != self.max_seq:
            old = np.linspace(0, 1, T)
            new = np.linspace(0, 1, self.max_seq)
            tmp = np.zeros((self.max_seq, 21, 3), dtype=feats.dtype)
            for v in range(21):
                for c in range(3):
                    tmp[:, v, c] = np.interp(new, old, feats[:, v, c])
            feats = tmp
        
        return feats  # (max_seq, 21, 3)

    def __len__(self):
        return len(self.samples)
    
   
    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        x = torch.from_numpy(feats).permute(2, 0, 1)  # [3, T, V]
        teacher = velocity_similarity_teacher(x.float(), tau=0.5, topk=4, remove_self=True) 

        A_temporal, A_temporal_scores = build_temporal_edges(
            self.max_seq,
            device=x.device,
            model=self.temporal_edge_model,  # may be None -> fixed rules
            x=x,
            threshold=self.edge_threshold,
             return_scores=True
        )
        A_spatial = IPN_AM.to(x.device)
        y = torch.tensor(label, dtype=torch.long)

        return {
            'Sequence': x,                   # [3, T, V]
            'A_spatial': A_spatial,          # [V, V]
            'A_temporal_teacher': teacher,   # [T, V, V] (soft probs), last frame zeros
            'A_temporal': A_temporal, 'A_temporal_scores': A_temporal_scores, 
            'Label': y
        }


if __name__ == '__main__':

    print('Pretraining dataset:')
    dataset = IPNDataset(data_dir="D:\\Dataset\\IPN\\ipn\\landmarks",
                         ann_file="D:\\Dataset\\IPN\\ipn\\annotations\\Annot_TrainList_splitted.txt" ,
                           normalize=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print('Batch[ "Sequence" ] shape:', batch['Sequence'].shape)
    print('Batch[ "A_temporal" ] shape:', batch['A_temporal'].shape)
    print('Batch[ "A_spatial" ] shape:', batch['A_spatial'].shape)


    print('\nFinetuning dataset:')
    dataset = FinetuningDataset(data_dir="D:\\Dataset\\IPN\\ipn\\landmarks", 
                                ann_file="D:\\Dataset\\IPN\\ipn\\annotations\\Annot_TrainList_splitted.txt" ,
                                
                               )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print('Sequence shape:', batch['Sequence'].shape)
    print('Batch[ "A_temporal" ] shape:', batch['A_temporal'].shape)
    print('Batch[ "A_spatial" ] shape:', batch['A_spatial'].shape)
    print('Labels        :', batch['Label'])
    