import torch
from random import shuffle
import anytree as at
from anytree.node import Node
from torch.utils.data import Dataset
from h5py import File
import os

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

class TreeDataset_HDF5(Dataset):
    """ Info """
    # max_depth specifies how deep into the tree data will be taken from. It must be at least 1.
    # If exclude_leaves is False, then fraction_bottom_nodes_to_keep specifies the fraction of nodes at the bottom level of the tree 
    # to keep in the dataset. The default is to keep all of them; however, it is recommended to keep significantly fewer than 100%
    # in order to balance out the number of those samples used in training with the number of samples from higher levels of the tree.
    # This is because each level has at least twice as many nodes as the previous level, depending on the choice of non_root_children
    # in get_data_random_branching.py
    def __init__(self, root_dict, tokens_file, max_depth=None, exclude_leaves=False, fraction_bottom_nodes_to_keep=1.0, objective=None,
                 pad_token_id=32000):
        assert isinstance(tokens_file, str) and tokens_file.endswith(".hdf5"), "tokens_file must be a path to an HDF5 file"
        assert objective is not None, "Objective must be specified to extract correct values from the nodes"
        self.tokens_file = File(tokens_file, "r")
        self.node_list = []
        # Add non-iterative data to the list of nodes/samples
        for prompt_name, root in root_dict.items():
            root.name = prompt_name
            assert type(root) is ValNodeTokensHDF5
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
            self.node_list = self.node_list + cur_node_list
        # print("Size of dataset:", len(self.node_list))
        self.objective = objective
        self.pad_token_id = pad_token_id
        shuffle(self.node_list)

    def __len__(self):
        return len(self.node_list)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        node = self.node_list[idx]
        if node.root.name.startswith("prompt"):
            dataset_name = node.root.name
            leaf_tokens_indx_path = node.get_tokens_indx_path()
            # Due to some bug, there is a chance for an IndexError here, so we catch it and return a different sample
            try:
                sequence_list = [torch.Tensor(self.tokens_file[dataset_name][indx]) for indx in leaf_tokens_indx_path]
            except IndexError:
                # Until the bug is fixed, just choose a different sample
                alternate_idx = (idx + 1) % len(self.node_list)
                return self.__getitem__(alternate_idx)
            sequence_list_unpadded = [seq[seq != self.pad_token_id].unsqueeze(0) for seq in sequence_list]
            node_sequence = torch.cat(sequence_list_unpadded, dim=1)
            if node_sequence.dtype != torch.int64:
                node_sequence = node_sequence.to(torch.int64)
        else:
            raise ValueError("Node root name does not start with 'prompt', which should not happen")
        value = node.value[self.objective] if type(node.value) is dict else node.value
        assert value is not None, "Got a node with no value; the data probably hasn't been labeled using label_tree.py"
        return {"input_ids": node_sequence.squeeze(), "attention_mask": torch.ones_like(node_sequence.squeeze()), "label": value, "depth": node.depth}

class TreeDataset_Soft(TreeDataset_HDF5):
    """ This is a dataset for training a soft value model. It is similar to TreeDataset_HDF5"""
    def __init__(self, root_dict, tokens_file, max_depth=None, exclude_leaves=False, fraction_bottom_nodes_to_keep=1.0, objective=None,
                 pad_token_id=32000, KL_penalty=None):
        super().__init__(root_dict, tokens_file, max_depth, exclude_leaves, fraction_bottom_nodes_to_keep, objective, pad_token_id)
        if KL_penalty is None:
            raise ValueError("KL_penalty must be specified for TreeDataset_Soft")
        self.KL_penalty = KL_penalty

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        node = self.node_list[idx]
        if node.root.name.startswith("prompt"):
            dataset_name = node.root.name
            leaf_tokens_indx_path = node.get_tokens_indx_path()
            try:
                sequence_list = [torch.Tensor(self.tokens_file[dataset_name][indx]) for indx in leaf_tokens_indx_path]
            except IndexError:
                # Until the bug is fixed, just choose a different sample
                alternate_idx = (idx + 1) % len(self.node_list)
                return self.__getitem__(alternate_idx)
            sequence_list_unpadded = [seq[seq != self.pad_token_id].unsqueeze(0) for seq in sequence_list]
            node_sequence = torch.cat(sequence_list_unpadded, dim=1)
            if node_sequence.dtype != torch.int64:
                node_sequence = node_sequence.to(torch.int64)
        value = node.value[self.objective] if type(node.value) is dict else node.value
        assert value is not None, "Got a node with no value; the data probably hasn't been labeled using label_tree.py"
        expected_log_prob_ratio = node.expected_log_prob_ratio
        assert expected_log_prob_ratio is not None, "Got a node with no expected_log_prob_ratio; the data probably hasn't been labeled using label_tree_soft.py"
        soft_value = value - self.KL_penalty * expected_log_prob_ratio
        return {"input_ids": node_sequence.squeeze(), "attention_mask": torch.ones_like(node_sequence.squeeze()), "label": soft_value, "depth": node.depth}

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
