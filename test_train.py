# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for trainer."""
import ray
import torch
from qiskit_ibm_runtime import QiskitRuntimeService
from ray import train
from ray.air import session
from torch import nn

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from quantum_serverless.train.trainer import (
    QiskitScalingConfig,
    QiskitTorchTrainer,
    get_runtime_session,
)

QISKIT_RUNTIME_ACCOUNT_CONFIG_KEY = "qiskit_runtime_account"
QISKIT_RUNTIME_BACKENDS_CONFIG_KEY = "qiskit_runtime_backends_configs"


INPUT_SIZE = 1
LAYER_SIZE = 2
OUTPUT_SIZE = 1
NUM_EPOCHS = 1

def create_qnn(session, layer_size):
    feature_map = ZZFeatureMap(layer_size)
    ansatz = RealAmplitudes(layer_size, reps=1)

    qc = QuantumCircuit(layer_size)
    qc.append(feature_map, range(layer_size))
    qc.append(ansatz, range(layer_size))

    qnn = EstimatorQNN(
        estimator=Estimator(session=session),
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True
    )
    return TorchConnector(qnn)

# pylint:disable=no-member
class NeuralNetwork(nn.Module):
    """Test neural network."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(INPUT_SIZE, LAYER_SIZE)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(LAYER_SIZE, OUTPUT_SIZE)

    def forward(self, input_tensor):
        """Forward pass."""
        return self.layer2(self.relu(self.layer1(input_tensor)))

# pylint:disable=no-member
class HybridQNN(nn.Module):
    """Test neural network."""

    def __init__(self, session):
        super().__init__()
        self.layer1 = nn.Linear(INPUT_SIZE, LAYER_SIZE)
        self.relu = nn.ReLU()
        self.qnn = create_qnn(session, LAYER_SIZE)

    def forward(self, input_tensor):
        """Forward pass."""
        x = self.relu(self.layer1(input_tensor))
        x = self.qnn(x)
        return x

def train_loop(config):
    """Test training loop."""
    runtime_session = get_runtime_session(config)
    print(runtime_session)

    is_qnn = config.get("is_qnn")
    dataset_shard = session.get_dataset_shard("train")
    loss_fn = nn.MSELoss()

    if is_qnn:
        model = HybridQNN(runtime_session)
    else:
        model = NeuralNetwork()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model = train.torch.prepare_model(model)

    for epoch in range(NUM_EPOCHS):
        for batches in dataset_shard.iter_torch_batches(
            batch_size=32, dtypes=torch.float
        ):
            inputs, labels = torch.unsqueeze(batches["x"], 1), batches["y"]
            output = model(inputs)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss.item()}")


if __name__ == "__main__":

    """Tests for trainer."""

    from tokens import Tokens
    API_TOKEN = Tokens.API_TOKEN
    QiskitRuntimeService.save_account(channel="ibm_quantum",
                                      token=API_TOKEN,
                                      overwrite=True)

    """Tests trainer."""
    train_dataset = ray.data.from_items(
        [{"x": x, "y": 2 * x + 1} for x in range(200)]
    )
    scaling_config = QiskitScalingConfig(num_workers=3, num_qubits=1)
    runtime_service = QiskitRuntimeService(channel="ibm_quantum")

    print("runtime service: ", runtime_service)
    trainer = QiskitTorchTrainer(
        train_loop_per_worker=train_loop,
        qiskit_runtime_service_account=runtime_service.active_account(),
        scaling_config=scaling_config,
        datasets={"train": train_dataset},
        # train_loop_config={"qiskit_runtime_account": runtime_service.active_account(),
        #                    "qiskit_runtime_backends_configs": "ibmq_qasm_simulator"},
        train_loop_config={"is_qnn": True},
    )
    trainer.fit()