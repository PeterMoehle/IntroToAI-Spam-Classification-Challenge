# Introduction to AI: Spam Classification Challenge

An implementation of a cute little Framework for a Spam Classification Challenge. Based on the module Introduction to AI.

>If, at any point before the written exam, you (as a group) submit an e-mail spam classifier that beats my precision/recall performance on the e-mail data test set, I will award everyone a
2.5% bonus
in the written exam (e.g., if you score 87.5% in the exam, I will bump it up to 90%).

---
### Before you start:
Unzip the file `data_train/original.zip`. You should then have two folders (`nospam`, `spam`) and a file named `vocabulary.npy` in this directory. 

---

## Rules
- Must be submitted before the day of the written exam.
- Must be implemented in Python.
- Must use a method discussed in the course (variations/modifications are okay).
- Must be a single submission that all of you unanimously agree on.
- You win and earn the 2.5% bonus, if 2*(precision*recall)/(precision+recall) of your model on the test set is
larger than mine (this is called F1-score).
- I will upload the training set and my own code on May 12, 2025. You won’t have access to the test set.

##  Project Structure

```
spam-classification-challenge/
├── data_train/                    # Training data directory
│   └── original/
│       ├── spam/                  # Spam email files
│       ├── nospam/               # Non-spam email files
│       └── vocabulary.npy        # Vocabulary file (optional)
├── models/                        # Model implementations
│   ├── __init__.py
│   ├── abc_model.py              # Abstract base model interface
│   ├── naive_bayes.py            # Naive Bayes implementation
│   └── neural_network.py         # Neural Network implementation
├── graph/                         # Computational graph framework
│   ├── __init__.py
│   ├── abc_node.py               # Abstract node base class
│   ├── abc_graph_model.py        # Abstract graph model base class
│   ├── abc_loss_graph.py         # Abstract loss graph base class
│   ├── demo/                     # Demo implementation
│   │   ├── graph_demo.py         # Implementation of Backprop-Exercise
│   │   ├── BackPropExcercise.dot # Demonstraction Visualization
│   ├── nodes/                    # Node implementations
│   │   ├── __init__.py
│   │   ├── function_node.py      # Function node implementation
│   │   └── value_node.py         # Value node implementation
│   ├── functions/                # Mathematical functions
│   │   ├── __init__.py
│   │   ├── arithmetic.py         # Arithmetic operations (Add, Multiply, DotProduct)
│   │   ├── activation.py         # Activation functions (ReLU, Sigmoid, Tanh)
│   │   └── math_functions.py     # Mathematical functions (Exp, Log)
│   └── loss/                     # Loss functions
│       ├── __init__.py
│       ├── loss_functions.py     # Loss implementations (MSE, Hinge, Logistic)
│       └── loss_factory.py       # Loss factory pattern
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and preprocessing
│   └── evaluator.py              # Model evaluation with K-fold CV
├── main.py                       # Main evaluation script
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Usage

**Run the main evaluation**:
```bash
python main.py
```

This will:
- Load and preprocess the email data
- Train multiple models using different configurations
- Perform k-fold cross-validation
- Display comprehensive performance metrics
- Identify the best performing model

---
**Run the computational graph demonstration**
```bash
python python/graph/demo/graph_demo.py
```

This will:
- Create a computational graph for backpropagation exercise (`backpropagation and convolutions.pdf`, page 26)
- Solve it for parameter w
- Display the computational steps
- Display and store the created digraph diagram

## Customization

### Adding New Models

1. **Create a new model class** inheriting from `Model`:
   ```python
   from models.abc_model import Model
   
   class MyModel(Model):
       def train(self, X_train, y_train, verbose=True):
           # Implement training logic
           pass
       
       def predict(self, X):
           # Implement prediction logic
           pass
       
       def get_binary_predictions(self, X, threshold=0.5):
           # Implement binary classification
           pass
   ```

2. **Add to main.py**:
   ```python
   from models.my_model import MyModel
   
   models.append(MyModel(name="My Custom Model"))
   ```

### Adding New Loss Functions

1. **Create a loss class** in `graph/loss/loss_functions.py`:
   ```python
   class Loss(LossGraph):
       def _build_loss(self) -> Node:
           # Build computation graph for your loss
           pass
   ```

2. **Update the factory** in `graph/loss/loss_factory.py`:
   ```python
   elif loss_type == "loss":
       loss = MyCustomLoss(prediction=output_node, target=target_node).get_loss_node()
   ```

### Adding New Activation Functions

1. **Create an activation class** in `graph/functions/activation.py`:
   ```python
   class MyActivation(Function):
       def compute(self, nodes):
           # Forward computation
           pass
       
       def derivate(self, node, nodes):
           # Backward computation
           pass
   ```
   
### Adding new Computational Graph

1. **Create a new model class** inheriting from `Model`:
   ```python
   class NewGraph(BaseGraph):
       def __init__(self):
   ```

2. **Add constants and computations** as seen in `/graph/demo/graph_demo.py`

3. **Initialize super constructor** with values **output_node** and **loss_node**

4. **Render it using a graphviz tool/extension of your choice** 