# Model Descriptions

This document explains the implemented models from the code up. Each model
section starts from the relevant code snippets, then explains what the code
computes and how it corresponds to the mathematical model.

All implemented models are supervised classifiers. They take an input example

```text
x = (x_1, ..., x_D)
```

and return one logit per class:

```text
z(x) = (z_1(x), ..., z_C(x)).
```

The logits are unnormalized class scores. The training wrapper converts them to
class probabilities through softmax:

```text
p(y = c | x) = exp(z_c(x)) / sum_k exp(z_k(x)).
```

For a training example with true label `y`, the loss is cross-entropy:

```text
L(x, y) = -log p(y | x).
```


## Logistic Regression

File: `src/models/logistic_regression.py`

### Layer

```python
self.linear = nn.Linear(input_dim, num_classes)
```

This creates a single affine map from the input vector to the class logits. The
layer stores a weight matrix and a bias vector:

```text
W in R^{C x D}
b in R^C
```

where:

- `D = input_dim`
- `C = num_classes`

For this model, the input uses the baseline representation. That means
categorical variables have already been one-hot encoded before the model sees
them.

### Forward Pass

```python
def forward(self, x):
    return self.linear(x)
```

The forward pass applies the affine map:

```text
z(x) = W x + b.
```

For a batch of `B` examples:

```text
x shape = B x D
z shape = B x C
```

The model does not apply softmax. This is intentional because
`CrossEntropyLoss` expects raw logits.


## Multilayer Perceptron

File: `src/models/mlp.py`

### Network

```python
self.network = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, num_classes),
)
```

This builds a feed-forward neural network with two hidden layers. The linear
layers learn affine maps, while the ReLU layers add nonlinearity.

The computation is:

```text
h_1 = ReLU(W_1 x + b_1)
h_2 = ReLU(W_2 h_1 + b_2)
z   = W_3 h_2 + b_3.
```

The hidden dimension controls the width of the two hidden layers:

```text
W_1 in R^{H x D}
W_2 in R^{H x H}
W_3 in R^{C x H}
```

where `H = hidden_dim`.

Dropout is applied after each hidden activation during training. It randomly
sets some hidden activations to zero, which reduces reliance on individual
hidden units.

### Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.network(x)
```

The forward pass applies the full sequential network and returns logits:

```text
x shape = B x D
z shape = B x C
```

Like logistic regression, the MLP uses the baseline input representation.


## CPD Classifier

File: `src/models/cpd.py`

The CPD classifier is a supervised tensor model for discrete inputs. It does
not model the input distribution. Instead, it directly parameterizes the class
logits.

### Feature Factors

```python
self.feature_factors = nn.ParameterList(
    [nn.Parameter(torch.empty(dim, rank)) for dim in feature_dims]
)
```

This creates one factor matrix per input feature. If feature `d` has `I_d`
possible values, its factor matrix has shape:

```text
A^{(d)} in R^{I_d x R}.
```

The rank `R` controls how many latent components the model uses. For a feature
value `x_d`, the model selects row `x_d` from the feature's factor matrix:

```text
A^{(d)}_{x_d,:}.
```

The CPD model uses the tensor input representation, so each feature value is an
integer index.

### Class Parameters

```python
self.class_weights = nn.Parameter(torch.empty(num_classes, rank))
self.class_bias = nn.Parameter(torch.empty(num_classes))
```

The class weights say how each rank component contributes to each class:

```text
lambda in R^{C x R}.
```

The class bias adds one free offset per class:

```text
b in R^C.
```

### Rank Products

```python
rank_components = torch.ones(
    x.size(0),
    self.rank,
    device=x.device,
)
```

The rank components start at one because the CPD interaction is a product over
features. For a batch of `B` examples:

```text
rank_components shape = B x R.
```

```python
for j, factor in enumerate(self.feature_factors):
    feature_values = x[:, j].long()
    rank_components = rank_components * factor[feature_values]
```

For each feature `j`, the code selects the factor row for every example in the
batch and multiplies it into the current rank components.

After all features are processed, each component is:

```text
q_r(x) = prod_d A^{(d)}_{x_d,r}.
```

### Logits

```python
return rank_components @ self.class_weights.T + self.class_bias
```

The rank components are mapped to class logits:

```text
z_c(x) = sum_r lambda_{c,r} q_r(x) + b_c.
```

Substituting the CPD product gives:

```text
z_c(x) = sum_r lambda_{c,r} prod_d A^{(d)}_{x_d,r} + b_c.
```


## MBA Classifier

File: `src/models/mba.py`

The MBA classifier is a supervised model for discrete inputs. It does not learn
a joint density over features and labels. Instead, it directly parameterizes one
class logit with interaction tables up to a chosen order.

### Interaction Sets

```python
for order in range(1, self.interaction_order + 1):
    for interaction in combinations(range(len(feature_dims)), order):
        dims = [feature_dims[index] for index in interaction]
```

This enumerates all feature subsets up to the maximum interaction order. If the
order is `K`, the model includes every subset `S` with:

```text
1 <= |S| <= K.
```

For example, order `1` uses only main effects. Order `2` uses main effects and
pairwise interactions. Order `3` also adds three-way interactions.

### Interaction Tables

```python
tables.append(
    nn.Parameter(
        torch.empty(num_classes, prod(dims))
    )
)
```

Each subset gets a class-specific table. If subset `S` contains features with
cardinalities `I_d`, the table has one entry for every value combination:

```text
theta^{(S)} in R^{C x prod_{d in S} I_d}.
```

This is the supervised “one MBA per class” form from the supervisor feedback:
the class dimension is part of every interaction table.

### Mixed-Radix Indexing

```python
def _make_strides(dims: list[int]) -> torch.Tensor:
    strides: list[int] = []
    current = 1
    for dim in reversed(dims):
        strides.append(current)
        current *= dim
    return torch.tensor(list(reversed(strides)), dtype=torch.long)
```

The table is stored as a flat vector, so a tuple of feature values must be
converted into a single index. For subset `S = (d_1, ..., d_k)`, the flat index
is:

```text
i_S(x) = sum_{m=1}^k x_{d_m} s_m,
```

where `s_m` is the product of the cardinalities after feature `d_m` in the
subset.

### Logits

```python
logits = self.class_bias.unsqueeze(0).expand(x.size(0), -1)

for idx, (interaction, table) in enumerate(
    zip(self.interactions, self.interaction_tables)
):
    values = x[:, interaction].long()
    strides = getattr(self, f"_stride_{idx}").to(x.device)
    flat_index = (values * strides).sum(dim=1)
    logits = logits + table[:, flat_index].T

return logits
```

The model starts with one bias per class and adds the selected interaction
score from every included subset. The resulting class logit is:

```text
z_c(x) = b_c + sum_{S: 1 <= |S| <= K} theta^{(S)}_{c, i_S(x)}.
```

The class probabilities are then:

```text
P(y = c | x) = exp(z_c(x)) / sum_{c'} exp(z_{c'}(x)).
```


## Tensor-Train Classifier

File: `src/models/tt.py`

The tensor-train classifier also works on discrete feature indices. Instead of
multiplying rank vectors elementwise like CPD, it contracts a chain of small
matrices.

### Feature Cores

```python
cores: list[nn.Parameter] = [
    nn.Parameter(torch.empty(feature_dims[0], 1, rank))
]
cores.extend(
    nn.Parameter(torch.empty(dim, rank, rank))
    for dim in feature_dims[1:]
)
self.feature_cores = nn.ParameterList(cores)
```

This creates one tensor-train core per feature. The first core has a left rank
of `1`, so selecting from it produces a row vector. Later cores produce
rank-by-rank matrices:

```text
G^{(1)} in R^{I_1 x 1 x R}
G^{(d)} in R^{I_d x R x R}, d > 1.
```

Each feature value selects one slice from its core.

### Class Parameters

```python
self.class_weights = nn.Parameter(torch.empty(rank, num_classes))
self.class_bias = nn.Parameter(torch.empty(num_classes))
```

After contracting the feature cores, the model has a rank vector. These
parameters map that rank vector to class logits:

```text
W in R^{R x C}
b in R^C.
```

### Chain Contraction

```python
state = self.feature_cores[0][x[:, 0].long()].squeeze(1)
```

This initializes the chain by selecting the first feature's core slice. For a
batch of `B` examples:

```text
state shape = B x R.
```

```python
for j, core in enumerate(self.feature_cores[1:], start=1):
    matrices = core[x[:, j].long()]
    state = torch.bmm(state.unsqueeze(1), matrices).squeeze(1)
```

For each remaining feature, the code selects a matrix for every example and
multiplies it into the current state.

The resulting TT state is:

```text
q(x) = G^{(1)}[x_1] G^{(2)}[x_2] ... G^{(D)}[x_D].
```

This contraction keeps the model compact because it never builds the full
feature interaction tensor explicitly.

### Logits

```python
return state @ self.class_weights + self.class_bias
```

The final rank vector is mapped to class logits:

```text
z(x) = q(x) W + b.
```


## Tensor-Ring Classifier

File: `src/models/tr.py`

The tensor-ring classifier is similar to the tensor-train classifier, but every
feature core is a rank-by-rank matrix. The contraction is closed as a ring
instead of left open as a chain.

### Feature Cores

```python
self.feature_cores = nn.ParameterList(
    [nn.Parameter(torch.empty(dim, rank, rank)) for dim in feature_dims]
)
```

Each feature has a table of matrices:

```text
G^{(d)} in R^{I_d x R x R}.
```

For a feature value `x_d`, the selected slice is:

```text
G^{(d)}[x_d] in R^{R x R}.
```

### Class Parameters

```python
self.class_matrices = nn.Parameter(
    torch.empty(num_classes, rank, rank)
)
self.class_bias = nn.Parameter(torch.empty(num_classes))
```

The ring is closed with a class-specific matrix. Each class has its own matrix:

```text
W_c in R^{R x R}.
```

The bias adds one scalar per class:

```text
b in R^C.
```

### Ring Contraction

```python
ring_state = self.feature_cores[0][x[:, 0].long()]
```

This selects the first feature matrix for each example:

```text
ring_state shape = B x R x R.
```

```python
for j, core in enumerate(self.feature_cores[1:], start=1):
    matrices = core[x[:, j].long()]
    ring_state = torch.bmm(ring_state, matrices)
```

The code multiplies the selected feature matrices:

```text
Q(x) = G^{(1)}[x_1] G^{(2)}[x_2] ... G^{(D)}[x_D].
```

### Logits

```python
logits = torch.einsum("bij,cji->bc", ring_state, self.class_matrices)
return logits + self.class_bias
```

The einsum contracts each example's ring state with every class matrix. This is
the trace-style ring closure:

```text
z_c(x) = trace(Q(x) W_c) + b_c.
```


## Training Wrapper

File: `src/models/lightning_module.py`

All model classes only define the forward pass. Training logic is shared by
`TabularClassifierModule`.

### Loss

```python
self.criterion = nn.CrossEntropyLoss()
```

This creates the loss function used for all models:

```text
L = (1 / B) sum_i -log p(y_i | x_i).
```

### Shared Step

```python
logits = self(x)
loss = self.criterion(logits, y)
preds = logits.argmax(dim=1)
acc = (preds == y).float().mean()
```

This computes logits, loss, predicted labels, and accuracy for one batch.

The prediction rule is:

```text
y_hat = argmax_c z_c(x).
```

Batch accuracy is:

```text
accuracy = (1 / B) sum_i 1[y_hat_i = y_i].
```

### Optimizer

```python
return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
```

This creates the Adam optimizer over all trainable parameters in the selected
model.


## Input Representations

The data module selects the representation from the model name:

```text
lr, mlp       -> baseline
cpd, mba, tt, tr -> tensor
```

The baseline representation is used by logistic regression and the MLP. It
one-hot encodes categorical features before the model sees them.

The tensor representation is used by CPD, MBA, TT, and TR. It keeps feature
values as integer indices so the models can perform factor, interaction-table,
or core lookups such as:

```text
A^{(d)}_{x_d,:}
G^{(d)}[x_d].
```
