import torch
from random import shuffle
import anytree as at
from anytree.node import Node
from torch.utils.data import Dataset
import pickle
from h5py import File
import os
from utils.hdf5_utils import load_all_pickled_objects, append_pickled_object, get_pickled_object

# NOTE: The has_eos attribute is not used anywhere, but I am keeping it to ensure compatibility with existing pickle files.
class ValNode(Node):
    def __init__(self, name, parent=None, children=None, value=None, has_eos=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.value = value
        self.has_eos = has_eos
        # Using this instead of the default "/" separator mean that walking the tree will cleanly concatenate the strings together,
        # allowing us to immediately tokenize the result
        self.separator = "" 

# This serves the same role as ValNodeTokens, but the tokens are stored in a separate hdf5 file and we only need to associate
# each node with an index into an array
class ValNodeTokensHDF5(ValNode):
    def __init__(self, tokens_indx, name=None, parent=None, children=None, value=None, log_prob_ratio=None, **kwargs):
        if name is None:
            name = ""
        super().__init__(name, parent, children, value, **kwargs)
        self.tokens_indx = tokens_indx
        self.log_prob_ratio = log_prob_ratio
        self.expected_log_prob_ratio = None  # This will be computed later

    def get_tokens_indx_path(self):
        return [n.tokens_indx for n in self.path]

# Function which retrieves the full sequence up to a given node in a tree of ValNodeTokensHDF5 nodes
# Note that to use this in TreeDataset_HDF5, the output will need to be unsqueezed to add a batch dimension
def get_sequence_for_node(node, tokens_file, pad_token_id, add_batch_dim=True):
    tokens_indx_path = node.get_tokens_indx_path()
    sequence_list = [torch.Tensor(tokens_file[node.root.name][indx]) for indx in tokens_indx_path]
    sequence_list_unpadded = [seq[seq != pad_token_id] for seq in sequence_list]
    node_sequence = torch.cat(sequence_list_unpadded, dim=0)
    if node_sequence.dtype != torch.int64:
        node_sequence = node_sequence.to(torch.int64)
    if add_batch_dim:
        node_sequence = node_sequence.unsqueeze(0)
    return node_sequence

# Function to load each leaf node's sequence from a dataset
# Includes an option to decode the sequences using a provided tokenizer (which will include special tokens)
def get_all_sequences_for_prompt(prompt_indx, hdf5_file, tokenizer, return_strings=False):
    dataset_name = f"prompt{prompt_indx}"
    assert dataset_name in hdf5_file.keys(), f"Dataset {dataset_name} not found in HDF5 file"
    root = get_pickled_object(hdf5_file, group_path="/trees", key=dataset_name)
    root.name = dataset_name
    sequences = []
    for node in at.PreOrderIter(root):
        if node.is_leaf:
            assert isinstance(node, ValNodeTokensHDF5), "Node is not of type ValNodeTokensHDF5"
            sequences.append(get_sequence_for_node(node, hdf5_file, tokenizer.pad_token_id, add_batch_dim=False))
    if return_strings:
        sequences = [tokenizer.decode(seq, skip_special_tokens=False) for seq in sequences]  
    return sequences

# A data structure which handles multiple sets of trees, where each set is paired with an hdf5 file
# This class is compatible with the new format where the hdf5 file contains both the tokens and the trees
class MixedTreeDataStructure:
    def __init__(self, hdf5_paths, num_val_trees):
        self.roots = {"train": [], "val": []}
        self.hdf5_files = []
        # Maps from root ID (integer) to a tuple (hdf5_indx, dataset_name)
        self.root_id_to_hdf5_dataset = {"train": {}, "val": {}}
        self.next_train_root_id = 0
        self.next_val_root_id = 0
        # Load each hdf5 file and the trees within it
        for hdf5_path in hdf5_paths:
            assert os.path.exists(hdf5_path), f"HDF5 file {hdf5_path} does not exist"
            hdf5_file = File(hdf5_path, "r")
            self.hdf5_files.append(hdf5_file)
            curr_trees = load_all_pickled_objects(hdf5_file, group_path="/trees")
            # Sort by prompt number
            sorted_tree_names = sorted(curr_trees.keys(), key=lambda x: int(x.replace("prompt", "")))
            if num_val_trees > 0 and len(self.roots["val"]) < num_val_trees:
                num_val_trees_to_add = min(num_val_trees - len(self.roots["val"]), len(curr_trees))
                val_tree_keys = sorted_tree_names[:num_val_trees_to_add]
                for i in range(num_val_trees_to_add):
                    root = curr_trees[val_tree_keys[i]]
                    root_id = self.next_val_root_id
                    self.next_val_root_id += 1
                    root.name = f"root{root_id}"
                    dataset_name = val_tree_keys[i]
                    assert dataset_name in hdf5_file.keys(), f"Dataset {dataset_name} not found in {hdf5_path}"
                    self.root_id_to_hdf5_dataset["val"][root_id] = (len(self.hdf5_files)-1, dataset_name)
                    self.roots["val"].append(root)
            else:
                num_val_trees_to_add = 0
            # The rest of the trees go to training
            train_tree_keys = sorted_tree_names[num_val_trees_to_add:]
            for i in range(len(train_tree_keys)):
                root = curr_trees[train_tree_keys[i]]
                root_id = self.next_train_root_id
                self.next_train_root_id += 1
                root.name = f"root{root_id}"
                dataset_name = train_tree_keys[i]
                assert dataset_name in hdf5_file.keys(), f"Dataset {dataset_name} not found in {hdf5_path}"
                self.root_id_to_hdf5_dataset["train"][root_id] = (len(self.hdf5_files)-1, dataset_name)
                self.roots["train"].append(root)

    def get_tokens_by_root_id(self, root_id, split):
        assert root_id in self.root_id_to_hdf5_dataset[split], f"Root ID {root_id} not found in mapping"
        hdf5_indx, dataset_name = self.root_id_to_hdf5_dataset[split][root_id]
        hdf5_file = self.hdf5_files[hdf5_indx]
        return torch.Tensor(hdf5_file[dataset_name])
    
    def get_node_list(self, split, exclude_leaves=False, max_depth=None, fraction_bottom_nodes_to_keep=1.0):
        node_list = []
        for root in self.roots[split]:
            if max_depth is None:
                max_depth = root.height
            if exclude_leaves:
                cur_node_list = [node for node in at.PreOrderIter(root) if node.depth <= max_depth and not node.is_leaf][1:]
            else:
                # Exclude the root node from this list
                upper_level_nodes = [node for node in at.PreOrderIter(root) if node.depth < max_depth][1:]
                bottom_level_nodes = [node for node in at.PreOrderIter(root) if node.depth == max_depth]
                # Remove bottom level nodes based on the fraction specified, but always keep at least one node
                num_bottom_nodes_to_keep = max(1, int(len(bottom_level_nodes) * fraction_bottom_nodes_to_keep))
                shuffle(bottom_level_nodes)
                bottom_level_nodes = bottom_level_nodes[:num_bottom_nodes_to_keep]
                cur_node_list = upper_level_nodes + bottom_level_nodes
            node_list = node_list + cur_node_list
        shuffle(node_list)
        return node_list

# In this case, the data structure is initialized with one or more paths to directories containing train/val folders
# and a hdf5 file. The point is to assign the root of each tree with a unique ID which can be used to
# identify the correct hdf5 files and which dataset within the hdf5 file to use when loading the tokens
# for that tree. The data structure maintains a mapping from each tree's root ID to its corresponding
# hdf5 file and dataset name.
class MixedTreeDataStructure_OldFormat(MixedTreeDataStructure):
    def __init__(self, dir_list, hdf5_filename="all_tokens.hdf5"):
        self.roots = {"train": [], "val": []}
        self.hdf5_files = []
        # Maps from root ID (integer) to a tuple (hdf5_indx, dataset_name)
        self.root_id_to_hdf5_dataset = {"train": {}, "val": {}}
        self.next_train_root_id = 0
        self.next_val_root_id = 0
        for d in dir_list:
            assert os.path.exists(d), f"Directory {d} does not exist"
            assert os.path.exists(os.path.join(d, hdf5_filename)), f"Directory {d} does not contain a '{hdf5_filename}' file"
            hdf5_file = File(os.path.join(d, hdf5_filename), "r")
            self.hdf5_files.append(hdf5_file)
            for split in ["train", "val"]:
                if os.path.exists(os.path.join(d, split)):  
                    split_dir = os.path.join(d, split)
                    for fname in os.listdir(split_dir):
                        if not fname.endswith(".pkl"):
                            continue
                        # We want to ensure that we can associate any node with its hdf5 dataset. It is easy to get from a node to its root,
                        # but the default name of the root may not be unique. So, we will assign each root a unique name based on an 
                        # incrementing integer ID.
                        with open(os.path.join(split_dir, fname), "rb") as f:
                            root = pickle.load(f)
                        assert type(root) is ValNodeTokensHDF5, "Tree root is not of type ValNodeTokensHDF5"
                        if split == "train":
                            root_id = self.next_train_root_id
                            self.next_train_root_id += 1
                        else:
                            root_id = self.next_val_root_id
                            self.next_val_root_id += 1
                        root.name = f"root{root_id}"
                        dataset_name = fname.split("_")[0]  # Will be something like "prompt1234"
                        assert dataset_name in hdf5_file.keys(), f"Dataset {dataset_name} not found in {hdf5_filename} in directory {d}"
                        self.root_id_to_hdf5_dataset[split][root_id] = (len(self.hdf5_files)-1, dataset_name)
                        self.roots[split].append(root)

class TreeDataset_HDF5(Dataset):
    """ Info """
    # max_depth specifies how deep into the tree data will be taken from. It must be at least 1.
    # If exclude_leaves is False, then fraction_bottom_nodes_to_keep specifies the fraction of nodes at the bottom level of the tree 
    # to keep in the dataset. The default is to keep all of them; however, it is recommended to keep significantly fewer than 100%
    # in order to balance out the number of those samples used in training with the number of samples from higher levels of the tree.
    # This is because each level has at least twice as many nodes as the previous level, depending on the choice of non_root_children
    # The KL_penalty argument causes the value of a node to be penalized based on the average average log probability ratio of the leaf
    # nodes below it. This is to encourage a lower KL divergence
    def __init__(self, root_dict, tokens_file, max_depth=None, exclude_leaves=False, fraction_bottom_nodes_to_keep=1.0, 
                 objective=None, pad_token_id=32000, KL_penalty=None):
        assert isinstance(tokens_file, str) and tokens_file.endswith(".hdf5"), "tokens_file must be a path to an HDF5 file"
        assert objective is not None, "Objective must be specified to extract correct values from the nodes"
        self.tokens_file = File(tokens_file, "r")
        self.node_list = []
        # Fill the list of nodes/samples
        for prompt_name, root in root_dict.items():
            root.name = prompt_name
            assert type(root) is ValNodeTokensHDF5
            curr_max_depth = root.height if max_depth is None else min(max_depth, root.height)
            if exclude_leaves:
                cur_node_list = [node for node in at.PreOrderIter(root) if node.depth <= curr_max_depth and not node.is_leaf][1:]
            else:
                # Exclude the root node from this list
                upper_level_nodes = [node for node in at.PreOrderIter(root) if node.depth < curr_max_depth][1:]
                bottom_level_nodes = [node for node in at.PreOrderIter(root) if node.depth == curr_max_depth]
                # Remove bottom level nodes based on the fraction specified, but always keep at least one node
                num_bottom_nodes_to_keep = max(1, int(len(bottom_level_nodes) * fraction_bottom_nodes_to_keep))
                shuffle(bottom_level_nodes)
                bottom_level_nodes = bottom_level_nodes[:num_bottom_nodes_to_keep]
                cur_node_list = upper_level_nodes + bottom_level_nodes
            self.node_list = self.node_list + cur_node_list
        # print("Size of dataset:", len(self.node_list))
        self.objective = objective
        self.pad_token_id = pad_token_id
        shuffle(self.node_list)
        self.KL_penalty = KL_penalty

    def __len__(self):
        return len(self.node_list)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        node = self.node_list[idx]
        if node.root.name.startswith("prompt"):
            # TODO: Replace this with a call to get_sequence_for_node; I need to make sure it won't
            # break anything first
            dataset_name = node.root.name
            leaf_tokens_indx_path = node.get_tokens_indx_path()
            sequence_list = [torch.Tensor(self.tokens_file[dataset_name][indx]) for indx in leaf_tokens_indx_path]
            sequence_list_unpadded = [seq[seq != self.pad_token_id].unsqueeze(0) for seq in sequence_list]
            node_sequence = torch.cat(sequence_list_unpadded, dim=1)
            if node_sequence.dtype != torch.int64:
                node_sequence = node_sequence.to(torch.int64)
        else:
            raise ValueError("Node root name does not start with 'prompt', which should not happen")
        value = node.value[self.objective] if type(node.value) is dict else node.value
        assert value is not None, "Got a node with no value; the data probably hasn't been labeled using label_tree.py"
        if self.KL_penalty is not None:
            expected_log_prob_ratio = node.expected_log_prob_ratio
            if expected_log_prob_ratio is None:
                # If the node is a leaf, then its expected log prob ratio is just its log prob ratio
                if node.is_leaf:
                    expected_log_prob_ratio = node.log_prob_ratio
            assert expected_log_prob_ratio is not None, "Got a node with no expected_log_prob_ratio; the data probably wasn't generated using get_data_iter.py"
            value = value - self.KL_penalty * expected_log_prob_ratio
        return {"input_ids": node_sequence.squeeze(), "attention_mask": torch.ones_like(node_sequence.squeeze()), "label": value, "depth": node.depth}

# Function to verify that a tree can be used for training
def check_tree(root, check_values=True, objectives=None, max_tokens_indx=None, check_probs=False):
    # Check that there are at least 3 layers
    if root.height < 2:
        print("Tree is too shallow")
        return False
    for node in at.PreOrderIter(root):
        if type(node) is not ValNodeTokensHDF5:
            if node.tokens_indx > max_tokens_indx:
                print("Node tokens index exceeds maximum index", max_tokens_indx)
                return False
        if node is root or node.is_leaf:
            continue
        if check_values:
            if node.value is None:
                print("Node has no value")
                return False
            elif objectives is not None:
                if isinstance(node.value, dict):
                    for objective in objectives:
                        if objective not in node.value.keys():
                            print("Node value does not contain objective", objective)
                            return False
                else:
                    print("Named objectives were specified, but node value is not a dictionary")
                    return False
        if check_probs:
            assert hasattr(node, "log_prob_ratio"), "Node does not have a log_prob_ratio attribute"
            if node.log_prob_ratio is None:
                print("Node does not have a log probability ratio")
                return False
            if not node.is_leaf:
                if node.expected_log_prob_ratio is None:
                    print("Node does not have an expected log probability ratio")
                    return False
    return True

# Converts a dataset in the old format (separate pickle files for each tree) to the new format
def convert_dataset_format(path_to_dataset, output_hdf5_filename="all_data_new_format.hdf5"):
    with File(os.path.join(path_to_dataset, output_hdf5_filename), "w") as out_hdf5_file:
        for split in ["train", "val"]:
            # Iterate over the pickle files and add them all to the hdf5 file using append_pickled_object
            split_dir = os.path.join(path_to_dataset, split)
            assert os.path.exists(split_dir), f"Directory {split_dir} does not exist"
            for fname in os.listdir(split_dir):
                if not fname.endswith(".pkl"):
                    continue
                with open(os.path.join(split_dir, fname), "rb") as f:
                    root = pickle.load(f)
                key = fname.split("_")[0]  # Will be something like "prompt1234"
                append_pickled_object(out_hdf5_file, key, root, group_path="/trees")
        # Also copy the tokens dataset to the new hdf5 file
        tokens_hdf5_path = os.path.join(path_to_dataset, "all_tokens.hdf5")
        assert os.path.exists(tokens_hdf5_path), f"Tokens HDF5 file {tokens_hdf5_path} does not exist"
        with File(tokens_hdf5_path, "r") as tokens_hdf5_file:
            for name in tokens_hdf5_file:
                out_hdf5_file.copy(tokens_hdf5_file[name], name)