import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import pytest
from sklearn.model_selection import train_test_split
from src.utils import load_config
from src.data_handling.gather_data import load_data, save_data, sample_data


# Define a fixture for temporary data
@pytest.fixture
def temp_data(tmpdir):
    temp_dir = tmpdir.mkdir("temp_data")
    csv_path = temp_dir.join("temp_data.csv")
    npy_path = temp_dir.join("temp_data.npy")
    npz_path = temp_dir.join("temp_data.npz")

    # Create sample data
    sample_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    sample_array = np.array([1, 2, 3])
    sample_sparse_matrix = csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    # Save sample data
    sample_df.to_csv(csv_path, index=False)
    np.save(npy_path, sample_array)
    save_npz(npz_path, sample_sparse_matrix)

    return {'csv': csv_path, 'npy': npy_path, 'npz': npz_path}

def test_load_data(temp_data):
    # Test loading CSV data
    csv_data = load_data(temp_data['csv'])
    assert isinstance(csv_data, pd.DataFrame)

    # Test loading NumPy data
    npy_data = load_data(temp_data['npy'])
    assert isinstance(npy_data, np.ndarray)

    # Test loading CSR matrix data
    npz_data = load_data(temp_data['npz'])
    assert isinstance(npz_data, csr_matrix)

def test_save_data(temp_data, tmpdir):
    # Test saving CSV data
    csv_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    csv_path = tmpdir.join("saved_data.csv")
    save_data(csv_data, str(csv_path))
    assert os.path.exists(str(csv_path))

    # Test saving NumPy data
    npy_data = np.array([1, 2, 3])
    npy_path = tmpdir.join("saved_data.npy")
    save_data(npy_data, str(npy_path))
    assert os.path.exists(str(npy_path))

    # Test saving CSR matrix data
    npz_data = csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    npz_path = tmpdir.join("saved_data.npz")
    save_data(npz_data, str(npz_path))
    assert os.path.exists(str(npz_path))

def test_sample_data():
    # Mock data
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'target': [0, 1, 0]})

    # Test stratified sampling
    sampled_data_stratified = sample_data(data, n_samples=2, target='target')
    assert isinstance(sampled_data_stratified, pd.DataFrame)
    assert len(sampled_data_stratified) == 2

    # Test random sampling
    sampled_data_random = sample_data(data, n_samples=2)
    assert isinstance(sampled_data_random, pd.DataFrame)
    assert len(sampled_data_random) == 2
