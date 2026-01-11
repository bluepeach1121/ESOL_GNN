# ESOL GNN

ESOL (“Delaney” dataset) is a MoleculeNet benchmark dataset for molecular property prediction where each molecule has a target aqueous solubility value (reported as log-scale solubility, logS). MoleculeNet is a collection of molecular datasets used to compare models across tasks like classification and regression. We can represent each molecule as a graph: atoms are nodes and chemical bonds are edges, perfect for Graph Neural Networks (GNNs). I used PyTorch Geometric (PyG) to handle graph batches, message-passing layers, and pooling, and RDKit to parse SMILES strings (text molecule descriptions) and convert them into molecule graphs. 
Note: The cell has plots but they aren't showing because of some problem with importing Jupyter cells that have outputs into GitHub. (11/1/2026)

---

## Models

Three 4-layer message-passing GNN regressors (GraphNorm + global mean pooling + MLP head):

- **GCN** --> *Graph Convolutional Network*
- **GINE** --> *Graph Isomorphism Network with Edge features* (GIN-E)
- **GATv2 (edge-aware)** --> *Graph Attention Network v2 with edge attributes in attention/message computation*

### Quick comparison

| Model | Uses edge attributes? | How neighbors are weighted | Message / aggregation | Strengths | Cost |
|------:|:----------------------:|----------------------------|----------------------------|----------|------|
| GCN   | No       | Fixed normalization from graph structure | Normalized sum/mean of neighbors | Simple, and strong baseline | Low |
| GINE  | Yes                    | Learned via MLP-style updates + edge terms | Sum aggregation + expressive update | Great for molecular graphs (bonds matter) | Medium |
| GATv2 (edge) | Yes            | Learned attention αᵢⱼ (depends on i, j, and edge) | Attention-weighted neighbor sum | Can focus on “important” neighbors | Higher |

---

## Message Passing

A standard message-passing layer updates node i by mixing its current features with aggregated neighbor messages:

$$
h_i^{(k+1)} = \phi\Big(\Lambda\big(h_i^{(k)}, \ \bigoplus_{j \in \mathcal{N}(i)} \mu_{ij}\big)\Big)
$$

- $h_i^{(k)}$: node i features at layer k
- $\mathcal{N}(i)$: 1-hop neighbors of i
- $\mu_{ij}$: “message” from neighbor j to i (model-dependent)
- $\bigoplus$: permutation-invariant aggregation (sum / mean / max)
- $\Lambda$: how self-info and neighbor-info are combined (often sum or concatenation)
- $\phi$: learnable update (often a linear/MLP + nonlinearity)

(message-passing equations and the convolutional / Message-passing / attentional categories are in the provided GNN intro PDF (check references).)

---

## 1) GCN -- Graph Convolutional Network

A common matrix form of a GCN layer:

$$
H^{(k+1)} = \sigma\Big(\tilde{D}^{-\frac12}\tilde{A}\tilde{D}^{-\frac12} H^{(k)} W^{(k)}\Big)
$$

Paper: SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS --> https://arxiv.org/pdf/1609.02907

**Variables**
- $H^{(k)} \in \mathbb{R}^{n \times d}$: node features for all n nodes at layer k
- $\tilde{A} = A + I$: the adjacency matrix of the undirected graph G with added self-connections.
- $\tilde{D}$: degree matrix of $\tilde{A}$
- $W^{(k)}$: trainable weight matrix
- $\sigma$: nonlinearity (e.g., ReLU). Each node becomes a normalized average of its neighbors’ features.

---

## 2) GINE -- Graph Isomorphism Network with Edge features (GINE)

A common GINE-style update:

$$
h_i^{(k+1)} = \text{MLP}^{(k)}\Big((1+\epsilon)\,h_i^{(k)} + \sum_{j\in \mathcal{N}(i)} \text{ReLU}\big(h_j^{(k)} + e_{ij}\big)\Big)
$$

Paper: HOW POWERFUL ARE GRAPH NEURAL NETWORKS? --> https://arxiv.org/pdf/1810.00826

**Variables**
- $h_i^{(k)}$: node i features at layer k
- $e_{ij}$: edge (bond) features between i and j
- $\epsilon$: scalar (fixed or learned) controlling how much “self” stays
- $\text{MLP}^{(k)}$: small neural net applied after aggregation. Neighbors contribute messages that explicitly include **bond information** ($e_{ij}$), important for chemistry.

---

## 3) Edge-aware GATv2 -- Graph Attention Network v2 (with edge attributes)

Attention-weighted aggregation:

$$
h_i^{(k+1)} = \sigma\Big(\sum_{j\in \mathcal{N}(i)} \alpha_{ij}\, W h_j^{(k)}\Big)
$$

Attention coefficients (edge-aware form):

$$
\alpha_{ij} = \text{softmax}_{j}\Big(\text{LeakyReLU}\big(a^\top [W h_i^{(k)} \,\Vert\, W h_j^{(k)} \,\Vert\, U e_{ij}]\big)\Big)
$$

Paper: How Attentive are Graph Attention Networks? --> https://arxiv.org/pdf/2105.14491

**Variables**
- $W$: linear projection for node features
- $U$: linear projection for edge features
- $a$: learned attention vector
- $[\cdot \Vert \cdot]$: concatenation
- $\text{softmax}_j$: normalize across neighbors of i so weights sum to 1

---

## Training + Evaluation Notes

- Task: graph regression (predict solubility from a molecule graph)
- Split: scaffold split 
- Loss: MSE
- Optimizer: AdamW + weight decay + gradient clipping

At the end, I included a **SMILES --> solubility predictor**:
1) Validate SMILES (parseable, non-empty, allowed elements, ≤ 200 atoms),
2) Convert to a PyG (pytorch geometric) graph,
3) Batch,
4) Run model forward pass,
5) Attach a confidence warning if molecule size is out-of-distribution relative to training.

---

## Results Summary

- **Best model:** GINE performed best overall (bonds explicitly help).
- **Error vs molecule size:** test error tended to increase with molecular size (# atoms).
- **Common issues encountered:**
  - Type/dtype errors (e.g., Float vs Long mismatches in graph tensors).
  - Normalization weirdness: adding normalization made training overfit and perform worse. I explored possible reasons in the .ipynb file.

---

## References
- “Introduction to Graph Neural Networks: A Starting Point for Machine Learning Engineers” --> https://arxiv.org/abs/2412.19419

- Pytorch Geometric --> https://pytorch-geometric.readthedocs.io/en/latest/
