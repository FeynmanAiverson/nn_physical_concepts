#   Copyright 2018 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import cPickle
import gzip
import io


def random_state(qubit_num):
    real = np.random.rand(2**qubit_num) - 0.5
    im = np.random.rand(2**qubit_num) - 0.5
    state = real + 1.j * im
    return state / np.linalg.norm(state)


def random_qubit_in_plane(state_num):

    def cart_to_spherical(xyz):
        xyz = np.array(xyz)
        xy = xyz[0]**2 + xyz[1]**2
        r = np.sqrt(xy + xyz[2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[2])
        phi = np.arctan2(xyz[1], xyz[0]) % (2 * np.pi)
        return r, theta, phi

    def bloch_to_qubit(theta, phi):
        return np.array([np.cos(theta / 2.) + 0.j, np.exp(1.j * phi) * np.sin(theta / 2.)])

    b1 = np.random.rand(3)
    b1 = b1 / np.linalg.norm(b1)
    b2 = np.random.rand(3)
    b2 = b2 - np.dot(b1, b2) * b1
    b2 = b2 / np.linalg.norm(b2)

    real_states = [a * b1 + b * b2 for a, b in (np.random.rand(state_num, 2) - 0.5)]
    theta_phi = [[cart_to_spherical(s)[1], cart_to_spherical(s)[2]] for s in real_states]
    complex_states = [bloch_to_qubit(*s) for s in theta_phi]

    return complex_states


def random_2_qubit_incomplete_projectors(n):
    # Generate 3 random 2-qubit states
    real = np.random.rand(3, 4) - 0.5
    im = np.random.rand(3, 4) - 0.5
    states = real + 1.j * im
    for i, s in enumerate(states):
        states[i] = s / np.linalg.norm(s)

    projector_basis = [s.conj().reshape([-1, 1]).dot(s.reshape([1, -1])) for s in states]

    projectors = []
    coeffs = np.random.rand(n, 3)
    for c in coeffs:
        projectors.append(c[0] * projector_basis[0] + c[1] * projector_basis[1] + c[2] * projector_basis[2])

    return projectors


def projection(a, b):
    return np.abs(np.dot(np.conj(a), b))**2


def measure(state, povm_element):
    res = state.conj().dot(povm_element).dot(state)
    assert(np.imag(res) < 1e-10)
    return np.real(res)


def create_data(qubit_num, measurement_num1, measurement_num2, sample_num, file_name=None, incomplete_tomography=False):
    """
    Params:
    qubit_num: number of qubits
    measurement_num1: number of projective measurements to be performed on input qubit
    measurement_num2: number of projective measurements to be performed on projection axis
    sample_num: number of training examples to be generated
    file_name: file is stored in /data/file_name.pkl.gz
    incomplete_tomography (only for single qubit): if False, a tomographically complete set of projectors is used for the input qubit
    """
    measurements = np.empty([sample_num, measurement_num1], dtype=np.float_)
    states = np.empty([sample_num, 2**qubit_num], dtype=np.complex_)
    if incomplete_tomography:
        projectors_in1 = random_qubit_in_plane(measurement_num1)
    else:
        projectors_in1 = [random_state(qubit_num) for _ in range(measurement_num1)]
    projectors_in2 = [random_state(qubit_num) for _ in range(measurement_num2)]
    input2 = np.empty([sample_num, measurement_num2])
    output = np.empty([sample_num, 1])
    for i in range(sample_num):
        sample = random_state(qubit_num)
        states[i] = sample
        measurements[i] = np.array([projection(p, sample) for p in projectors_in1])
        input2_state = random_state(qubit_num)
        input2[i] = np.array([projection(p, input2_state) for p in projectors_in2])
        output[i, 0] = projection(input2_state, sample)
    result = ([measurements, input2, output], states, [projectors_in1, projectors_in2])
    if file_name is not None:
        f = gzip.open(io.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result


def create_two_qubit_incomplete_tom(measurement_num1, measurement_num2, sample_num, file_name):
    qubit_num = 2
    measurements = np.empty([sample_num, measurement_num1], dtype=np.float_)
    states = np.empty([sample_num, 4], dtype=np.complex_)
    projector_matrices_in1 = random_2_qubit_incomplete_projectors(measurement_num1)
    projectors_in2 = [random_state(qubit_num) for _ in range(measurement_num2)]
    
    input2 = np.empty([sample_num, measurement_num2])
    output = np.empty([sample_num, 1])
    for i in range(sample_num):
        sample = random_state(qubit_num)
        states[i] = sample
        measurements[i] = np.array([measure(sample, p) for p in projector_matrices_in1])
        input2_state = random_state(qubit_num)
        input2[i] = np.array([projection(p, input2_state) for p in projectors_in2])
        output[i, 0] = projection(input2_state, sample)
    result = ([measurements, input2, output], states, [projector_matrices_in1, projectors_in2])
    if file_name is not None:
        f = gzip.open(io.data_path + file_name + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return result
