import os
import numpy as np
import torch
import tqdm
import os.path as opt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from utils.graph import build_temporal_edges


# 21‐joint hand connectivity (MediaPipe style)
# HAND_CONNECTIONS = [
#     (0,1),(1,2),(2,3),(3,4),(4,5),
#     (1,6),(6,7),(7,8),(8,9),
#     (1,10),(10,11),(11,12),(12,13),
#     (1,14),(14,15),(15,16),(16,17),
#     (1,18),(18,19),(19,20)
# ]


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

# Sanity‑check: every joint is present
assert set(_TEMPORAL_RULES.keys()) == set(range(21)), "Rules must cover 0‑20 joints"


# ------------------------------------------------------------------
# 2. Graph builder
# ------------------------------------------------------------------
def build_temporal_edges(T: int, device="cpu") -> torch.Tensor:
    """
    Create a (T, V, V) boolean tensor where entry [t, i, j] == 1
    means there is an edge *from* joint i at frame t
    *to*   joint j at frame t+1.

    The last frame has no outgoing links (all zeros).

    Parameters
    ----------
    T : int
        Number of frames in the sequence.
    device : str or torch.device
        Where to place the tensor.

    Returns
    -------
    A_tmp : torch.BoolTensor
        Shape = (T, 21, 21)
    """
    V = 21
    A_tmp = torch.zeros((T, V, V), dtype=torch.bool, device=device)

    # For frames 0 … T‑2 : add the user‑defined edges to next frame
    for t in range(T - 1):
        for src, dest_list in _TEMPORAL_RULES.items():
            for dst in dest_list:
                A_tmp[t, src, dst] = True

    # (Optional) self‑loop from each joint t→t+1  — already included for src==dst (rule 0)
    # If you *don’t* want identity edges, remove the first element (=src) from each dest_list above.

    return A_tmp

def get_adjacency_matrix(connections):
    n_lands = len(set([joint for tup in connections for joint in tup]))
    adj = np.zeros((n_lands, n_lands))
    for i, j in connections:
        adj[i][j] = 1.
        adj[j][i] = 1.
    adj += np.eye(adj.shape[0])
    return adj


IPN_AM = get_adjacency_matrix(HAND_CONNECTIONS)
IPN_AM = torch.from_numpy(IPN_AM).float()


class IPNDataset(Dataset):
    """
    Reads IPN annotations (.csv) and landmark .txt files.
    Returns: {'Sequence': Tensor(C=5, T=max_seq, V=21), 'Label': LongTensor}
    """
    def __init__(self, data_dir, ann_file, max_seq=4, normalize=True):
        self.data_dir  = data_dir
        self.ann_file  = ann_file
        self.max_seq   = max_seq
        self.normalize = normalize

        assert opt.exists(data_dir), 'path {} does not exist'.format(data_dir)
        ## data
        self.data_dir = data_dir
        
        self.label_map = ["no_gesture",
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

        # precompute connectivity count per joint
        self.connectivity = {i:0 for i in range(21)}
        for p,c in HAND_CONNECTIONS:
            self.connectivity[p] += 1
            self.connectivity[c] += 1


        # load all (feature, label) samples
        self.samples = []
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

    # def _load_file(self, path):
    #     if not os.path.exists(path): 
    #         return None
    #     # IPN: one frame per line, 21 joints × 3 floats each
    #     coords = np.loadtxt(path, delimiter=' ', dtype=np.float32)  # shape = (T_raw, 63)
    #     if coords.ndim != 2 or coords.shape[1] != 63:
    #         return None
    #     frames = coords.reshape(-1, 21, 3)  # (T_raw, 21, 3)
    #     if frames.shape[0] < 2:
    #         return None
    #     return self._post_process(frames)

    def _post_process(self, seq):
        # seq: (T, 21, 5)
        feats = seq  # keep only [x,y,z,vis,conn]

        # normalize XYZ if desired
        if self.normalize:
            mask  = feats[..., :3] != -1.0
            valid = feats[..., :3][mask]
            if valid.size:
                m, s = valid.mean(), valid.std()
                feats[..., :3] = (feats[..., :3] - m) / (s + 1e-6)

        T = feats.shape[0]
        # top‐k by movement magnitude if too long
        if T > self.max_seq:
            delta = np.linalg.norm(seq[1:,...,:3] - seq[:-1,...,:3], axis=(1,2))
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

        return feats  # shape = (max_seq,21,5)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # feats, label = self.samples[idx]
        feats = self.samples[idx]
        x = torch.from_numpy(feats).permute(2, 0, 1)  # → (C=5, T, V)
        # A = build_temporal_edges(self.max_seq, device=x.device)
        # # y = torch.tensor(label, dtype=torch.long)
        # # return {'Sequence': x, 'Label': y}
        # return {'Sequence': x, 'AM': A}
        A_temporal = build_temporal_edges(self.max_seq, device=x.device)
        A_spatial  = IPN_AM.to(x.device)                  # (V, V)
        # we’ll broadcast spatial across time or batch in the model
        return {
           'Sequence': x,
           'A_temporal': A_temporal,                      # (T, V, V)
           'A_spatial' : A_spatial                        # (V, V)
        }
    

    ####################################333333

class FinetuningDataset(Dataset):
    def __init__(self, data_dir,ann_file,max_seq=80, normalize=False):
        
        assert opt.exists(data_dir), 'path {} does not exist'.format(data_dir)
        ## data
        self.data_dir  = data_dir
        self.ann_file  = ann_file
        self.normalize = normalize
        self.max_seq = max_seq
        
        self.label_map = ["no_gesture",
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
        # seq: (T, 21, 5)
        feats = seq  # keep only [x,y,z,vis,conn]

        # normalize XYZ if desired
        if self.normalize:
            mask  = feats[..., :3] != -1.0
            valid = feats[..., :3][mask]
            if valid.size:
                m, s = valid.mean(), valid.std()
                feats[..., :3] = (feats[..., :3] - m) / (s + 1e-6)

        T = feats.shape[0]
        # top‐k by movement magnitude if too long
        if T > self.max_seq:
            delta = np.linalg.norm(seq[1:,...,:3] - seq[:-1,...,:3], axis=(1,2))
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

        return feats  # shape = (max_seq,21,5)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        x = torch.from_numpy(feats).permute(2, 0, 1)  # → (C=5, T, V)
        # A = build_temporal_edges(self.max_seq, device=x.device) 
        # y = torch.tensor(label, dtype=torch.long)
        # return {'Sequence': x, 'AM': A, 'Label': y}
        A_temporal = build_temporal_edges(self.max_seq, device=x.device)
        A_spatial  = IPN_AM.to(x.device)
        y = torch.tensor(label, dtype=torch.long)
        return {
          'Sequence'  : x,
          'A_temporal': A_temporal,
          'A_spatial' : A_spatial,
          'Label'     : y
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
                                normalize=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print('Sequence shape:', batch['Sequence'].shape)
    # print('Batch[ "A_temporal" ] shape:', batch['A_temporal'].shape)
    print('Batch[ "A_spatial" ] shape:', batch['A_spatial'].shape)
    print('Labels        :', batch['Label'])
    