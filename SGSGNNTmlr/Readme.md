# Instruction

The followings are for Gilbreth command

conda init bash
source ~/.bashrc


conda activate /home/das90/.conda/envs/cent7/2020.11-py38/py311cu117pyg200
export PATH=/home/das90/.conda/envs/cent7/2020.11-py38/py311cu117pyg200/bin:$PATH

cd GNNcodes/CVE2020/GNN-NC/Graph-Sparsification/SupervisedSparsification/SGSGNNTmlr/

python main.py --dataset SmallCora --mode learned --runs 1 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GCN --log True --sparse_edge_mlp True --conditional True --reg1 True --reg2 True

Inputs
  data.x, batch.edge_index
        |
        |  (no grad)
        v
edge_prob_mlp (full edges) ----------------------------> edge_probs_full (detached)
        |                                                |
        |                                                v
        |                                         gumbel_softmax_sampling
        |                                                |
        |                                                v
        |                                      sampled_edge_index
        |                                                |
        |  (grad enabled)                                |
        v                                                |
edge_prob_mlp (sampled edges) -> edge_probs_sampled ----+
        |
        v
GNN (EdgeProbGCN) on sampled_edge_index + edge_probs_sampled
        |
        v
learned_out
        |
        +-----------------------------+
        |                             |
     reg1 (sampled edges)          reg2 (sampled edges)
        |                             |
        +-------------+---------------+
                      |
                   total loss
                      |
                      v
           backprop to:
             - edge_prob_mlp (sampled edges only)
             - GNN parameters



flowchart TD
    A["data.x + batch.edge_index"] --> B["edge_prob_mlp (full edges)"]
    B -->|no_grad| C["edge_probs_full"]
    C --> D["gumbel_softmax_sampling"]
    D --> E["sampled_edge_index"]

    A --> F["edge_prob_mlp (sampled edges)"]
    E --> F
    F --> G["edge_probs_sampled"]

    A --> H["GNN (EdgeProbGCN)"]
    E --> H
    G --> H
    H --> I["learned_out"]

    I --> J["reg1 on sampled edges"]
    I --> K["reg2 consistency loss"]
    J --> L["total loss"]
    K --> L
    L --> M["backprop"]

    M --> F
    M --> H
