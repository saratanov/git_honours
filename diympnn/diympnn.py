#chemprop copy - a DIY message passing algorithm

#################################################
### args.py

class TrainArgs():
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""
    # Common arguments
    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    atom_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    bond_features_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""
    no_cache_mol: bool = False
    """
    Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    """
    
    # General arguments
    data_path: str
    """Path to data CSV file."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    ignore_columns: List[str] = None
    """Name of the columns to ignore when :code:`target_columns` is not provided."""
    dataset_type: Literal['regression', 'classification', 'multiclass'] = 'regression'
        #SARA EDITS
    """Type of dataset. This determines the loss function used during training."""
    multiclass_num_classes: int = 3
    """Number of classes when running multiclass classification."""
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    split_type: Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'cv', 'cv-no-test', 'index_predetermined'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    """Split proportions for train/validation/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    folds_file: str = None
    """Optional file of fold labels."""
    val_fold_index: int = None
    """Which fold to use as val for leave-one-out cross val."""
    test_fold_index: int = None
    """Which fold to use as test for leave-one-out cross val."""
    crossval_index_dir: str = None
    """Directory in which to find cross validation index files."""
    crossval_index_file: str = None
    """Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    metric: Metric = None
    """
    Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "auc" for classification and "rmse" for regression.
    """
    extra_metrics: List[Metric] = []
    """Additional metrics to use to evaluate the model. Not used for early stopping."""
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    Use "inf" to always cache.
    """
    save_preds: bool = False
    """Whether to save test split predictions during training."""

    # Model arguments
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    mpn_shared: bool = False
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0.0
    """Dropout probability."""
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    """Activation function."""
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    ffn_hidden_size: int = None
    """Hidden dim for higher-capacity FFN (defaults to hidden_size)."""
    ffn_num_layers: int = 2
    """Number of layers in FFN after MPN encoding."""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    separate_val_features_path: List[str] = None
    """Path to file with features for separate val set."""
    separate_test_features_path: List[str] = None
    """Path to file with features for separate test set."""
    separate_val_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_bond_features_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_bond_features_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""
    aggregation: Literal['mean', 'sum', 'norm'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""

    # Training arguments
    epochs: int = 30
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    """Trains with an equal number of positives and negatives in each batch."""

    overwrite_default_atom_features: bool = False
    """
    Overwrites the default atom descriptors with the new ones instead of concatenating them.
    Can only be used if atom_descriptors are used as a feature.
    """
    no_atom_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    overwrite_default_bond_features: bool = False
    """Overwrites the default atom descriptors with the new ones instead of concatenating them"""
    no_bond_features_scaling: bool = False
    """Turn off atom feature scaling."""

#################################################
### mpn.py

from typing import List, Union
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: TrainArgs object containing arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout_layer(atom_hiddens)                             # num_atoms x (hidden + descriptor size)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if args.mpn_shared:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                          for _ in range(args.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch[0]) != BatchMolGraph:
            # TODO: handle atom_descriptors_batch with multiple molecules per input
            if self.atom_descriptors == 'feature':
                if len(batch[0]) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [mol2graph(b, atom_features_batch, bond_features_batch,
                                   overwrite_default_atom_features=self.overwrite_default_atom_features,
                                   overwrite_default_bond_features=self.overwrite_default_bond_features)
                         for b in batch]
            elif bond_features_batch is not None:
                if len(batch[0]) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [mol2graph(b, None, bond_features_batch,
                                   overwrite_default_atom_features=self.overwrite_default_atom_features,
                                   overwrite_default_bond_features=self.overwrite_default_bond_features)
                         for b in batch]
            else:
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, atom_descriptors_batch) for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return output
