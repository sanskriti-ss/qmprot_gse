# ABOUTME: Shares a prepared analytic density matrix among groups measuring the same noisy circuit.
# ABOUTME: Keeps each group's original basis rotations, readout operations and measurement order.
import numpy as np
import pennylane as qml
from pennylane.devices import DefaultMixed


def _readout_order(circuits):
    """Place equal rotation prefixes together while keeping the terminal device state observable."""
    first = circuits[0].operations
    for op in first:
        if any(type(x) not in (float, int, complex) and not isinstance(x, np.number)
               and not (type(x) is np.ndarray and x.dtype.kind in "biufc")
               for x in op.data):
            return None
    keys = []
    for tape in circuits:
        if len(tape.operations) != len(first) or any(a is not b for a, b in zip(first, tape.operations)):
            return None
        for measurement in tape.measurements:
            if type(measurement) is not qml.measurements.ExpectationMP:
                return None
            obs = measurement.obs
            factors = obs.operands if type(obs) is qml.ops.Prod else (obs,)
            if any(type(p) not in (qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity) for p in factors):
                return None
        rotations = tape.diagonalizing_gates
        if any(type(op) not in (qml.Hadamard, qml.PauliZ, qml.S) for op in rotations):
            return None
        keys.append(tuple((op.name, op.wires[0]) for op in rotations))
    return sorted(range(len(circuits) - 1), key=keys.__getitem__) + [len(circuits) - 1]


class GroupedMixedState(DefaultMixed):
    def batch_execute(self, circuits, **kwargs):
        self._grouped = len(circuits) > 1 and not self.shots and not self._debugger
        self._group_operations = None
        self._readout_states = None
        try:
            if (self._grouped and len(circuits) >= 32 and not self.tracker.active and not self.readout_err
                    and tuple(self.wires) == tuple(range(self.num_wires)) and self.num_wires <= 10):
                order = _readout_order(circuits)
                if order is not None:
                    self._readout_states = []
                    self._state_limit = (32 * 1024**2) // (4**self.num_wires * np.dtype(self.C_DTYPE).itemsize)
                    values = super().batch_execute([circuits[i] for i in order], **kwargs)
                    results = [None] * len(order)
                    for index, value in zip(order, values, strict=True):
                        results[index] = value
                    return results
            return super().batch_execute(circuits, **kwargs)
        finally:
            self._group_operations = self._group_state = self._readout_states = None
            self._grouped = False

    def apply(self, operations, rotations=None, **kwargs):
        if not getattr(self, "_grouped", False):
            return super().apply(operations, rotations=rotations, **kwargs)
        if self._readout_states is not None:
            if self._group_operations is None:
                super().apply(operations, **kwargs)
                self._group_operations = tuple(operations)
                self._group_state = self._pre_rotated_state
            self._state = self._pre_rotated_state = self._group_state
            rotations = rotations or ()
            # Each saved density matrix follows the same native operations from the same preparation.
            shared = 0
            for op, (key, value) in zip(rotations, self._readout_states):
                if (type(op), op.wires) != key:
                    break
                self._state = value
                shared += 1
            del self._readout_states[shared:]
            for op in rotations[shared:]:
                self._apply_operation(op)
                if len(self._readout_states) < self._state_limit:
                    self._readout_states.append(((type(op), op.wires), self._state))
            return
        previous = self._group_operations
        if (previous is not None and len(previous) == len(operations)
                and all(a is b for a, b in zip(previous, operations))):
            self._state = self._group_state
            return super().apply((), rotations=rotations, **kwargs)
        self._group_state = None
        super().apply(operations, rotations=rotations, **kwargs)
        self._group_operations = tuple(operations)
        self._group_state = self._pre_rotated_state
