#using chemprop MolGraph feauturisation as a basis for an MPNN
#28th July

#TODO: design training loader to batch lists of smiles
#TODO: crossval
#TODO: testing metrics

###############
#FEATURISATION#
###############

from typing import List, Tuple, Union
from itertools import zip_longest
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
EXTRA_ATOM_FDIM = 0
BOND_FDIM = 14
EXTRA_BOND_FDIM = 0


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :return: The dimensionality of the atom feature vector.
    """
    return ATOM_FDIM + EXTRA_ATOM_FDIM


def get_bond_fdim(atom_messages: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :return: The dimensionality of the bond feature vector.
    """

    return BOND_FDIM + EXTRA_BOND_FDIM + (not atom_messages) * get_atom_fdim()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, mol: Union[str, Chem.Mol]):
        """
        :param mol: A SMILES or an RDKit molecule.
        """
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.a2a = None
        # Get atom features
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]

        self.n_atoms = len(self.f_atoms)

        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
                
    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Returns the components of the :class:`MolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[-get_bond_fdim(atom_messages=atom_messages):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb
    
    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = [[self.b2a[x] for x in y] for y in self.a2b]  # num_atoms x max_num_bonds

        return self.a2a
    

class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(mols: Union[List[str], List[Chem.Mol]]) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol) for mol in mols])

##############
### MODELS ###
##############

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Helper function.
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.

    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


class MPNN(nn.Module):
    """
    Model wrapper for message passing with a single molecule. Contains message passing featuriser + linear NN layer.
    """
    def __init__(self, args):
        super(MPNN, self).__init__()
        self.atom_fdim = get_atom_fdim()
        self.atom_messages = args.atom_messages
        self.bond_fdim = get_bond_fdim(atom_messages=args.atom_messages)
        self.hidden_size = args.hidden_size
        
        self.encoder = MP(args)
        self.NN = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, batch):
        """
        :param batch: list of SMILES strings
        """
        encodings = self.encoder(batch)
        output = self.NN(encodings)
        return output
    
class double_MPNN(nn.Module):
    """
    Model wrapper for message passing with two input molecules. Contains message passing featuriser + 2 linear NN layers.
    """
    def __init__(self, args):
        super(double_MPNN, self).__init__()
        self.atom_fdim = get_atom_fdim()
        self.atom_messages = args.atom_messages
        self.bond_fdim = get_bond_fdim(atom_messages=args.atom_messages)
        self.hidden_size = args.hidden_size
        self.interaction = args.interaction
        self.readout = args.readout
        
        # separate encoders for solute and solvent
        self.encoder_sol = MP(args)
        self.encoder_solv = MP(args)
        
        self.NN1 = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
        self.NN2 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, batch):
        """
        :param batch: List[Tuple[str],Tuple[str]]
                      list of 2 tuples containing solute / solvent SMILES strings
        """
        num_pairs = len(batch[0])
        sol = batch[0]
        solv = batch[1]
        
        # message passing (returns either a tensor of molecular or atomic feature vectors depending on interaction)
        sol = self.encoder_sol(sol)
        solv = self.encoder_solv(solv)
        
        # interaction step
        if self.interaction:
            # create interaction map between solute and solvent atomic feature vectors
            int_map = [torch.mm(sol[x],solv[x].t()) for x in range(num_pairs)]
            int_map = [torch.tanh(int_map[x]) for x in range(num_pairs)]
            
            # create molecular context
            sol_context = [torch.mm(int_map[x], solv[x]) for x in range(num_pairs)]
            solv_context = [torch.mm(int_map[x].t(), sol[x]) for x in range(num_pairs)] # num_pairs x (num_atoms x hidden_size)
            
            # concatenate molecule + context
            sol = [torch.cat((sol[x],sol_context[x])) for x in range(num_pairs)]
            solv = [torch.cat((solv[x],solv_context[x])) for x in range(num_pairs)] # num_pairs x (2*num_atoms x hidden_size)
            
            # pooling along the atomic dimension -> molecular feature vector
            if self.readout == 'mean':
                sol = torch.stack([torch.mean(mol, dim=0) for mol in sol])
                solv = torch.stack([torch.mean(mol, dim=0) for mol in solv])
            elif self.readout == 'sum':
                sol = torch.stack([torch.sum(mol, dim=0) for mol in sol])
                solv = torch.stack([torch.sum(mol, dim=0) for mol in solv])  # num_pairs x hidden_size
        
        # concatenate solute / solvent feature vectos
        encodings = torch.cat([sol,solv], dim=1) # num_pairs x 2*hidden_size
        
        # NN
        output = F.relu(self.NN1(encodings))
        output = F.relu(self.NN2(output))
        return output
    
class MP(nn.Module):
    """
    Message passing module based on chemprop. Can pass messages between atoms (MPNN) or directed bonds (D-MPNN).
    """
    def __init__(self, args):
        super(MP, self).__init__()
        self.atom_fdim = get_atom_fdim()
        self.atom_messages = args.atom_messages
        self.bond_fdim = get_bond_fdim(atom_messages=args.atom_messages)
        self.hidden_size = args.hidden_size
        self.depth = args.depth
        self.readout = args.readout
        self.dropout = args.dropout
        self.interaction = args.interaction
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, smiles):
        """
        :param smiles: list of SMILES strings in a batch
        
        :return mol_tensors: list of torch.Tensors containing atomic feature vectors for each mol in the batch
                             (for use with an interaction layer)
        :return mol_vecs: torch.Tensor of the final feature vectors for each mol in the batch
        """
        # convert list of smiles into BatchMolGraph
        mol = mol2graph(smiles)
        
        # get all feature vectors and connections
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol.get_components(atom_messages=self.atom_messages)
        if self.atom_messages:
            a2a = mol.get_a2a()

        # hidden state initialisation
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = F.relu(input)  # num_bonds x hidden_size
        
        # message passing
        for depth in range(self.depth - 1):
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
            message = F.relu(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden
        
        # last message passing step (same for MP and D-MP)
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = F.relu(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        if self.interaction:
            # collate atomic feature vectors in a tensor for each molecule
            mol_tensors = []
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    mol_tensors.append(self.cached_zero_vector)
                else:
                    cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                    mol_tensors.append(cur_hiddens)
            return mol_tensors # num_molecules x (num_atoms x hidden_size)
            
        else:
            # readout into a molecular feature vector
            mol_vecs = []
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    mol_vecs.append(self.cached_zero_vector)
                else:
                    cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                    mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                    if self.readout == 'mean':
                        mol_vec = mol_vec.sum(dim=0) / a_size
                    elif self.readout == 'sum':
                        mol_vec = mol_vec.sum(dim=0)
                    mol_vecs.append(mol_vec)
            mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
            return mol_vecs  # num_molecules x hidden

        
class TrainArgs:
    """
    Object containing arguments used for training a model.
    """
    depth = 3
    """Number of message passing steps"""
    hidden_size = 128
    """Size of final hidden feature vector"""
    dropout = 0.2
    """Dropout probability"""
    atom_messages = True
    """Centers messages of atoms (True) or bonds (False)"""
    readout = 'mean'
    """Readout function ('mean' or 'sum')"""
    interaction = False
    """Interaction layer between solute and solvent before creating final molecular feature vector (CIGIN model)"""