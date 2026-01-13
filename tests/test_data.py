from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import Dataset

from ml_ops_project.data import MyDataset, normalize, corrupt_mnist


def test_normalize():
    """Test the normalize function."""
    images = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    normalized = normalize(images)
    assert normalized.mean().item() < 1e-6  # Mean should be close to 0
    assert abs(normalized.std().item() - 1.0) < 1e-6  # Std should be close to 1


@patch('torch.load')
def test_my_dataset(mock_load):
    """Test the MyDataset class."""
    # Mock the torch.load to return dummy data
    mock_images = torch.randn(100, 28, 28)
    mock_targets = torch.randint(0, 10, (100,))
    mock_load.return_value = mock_images if "images" in str(mock_load.call_args) else mock_targets

    # Create alternating return values for images and targets
    mock_load.side_effect = [
        torch.randn(10, 28, 28),  # train_images_0
        torch.randint(0, 10, (10,)),  # train_target_0
        torch.randn(10, 28, 28),  # train_images_1
        torch.randint(0, 10, (10,)),  # train_target_1
        torch.randn(10, 28, 28),  # train_images_2
        torch.randint(0, 10, (10,)),  # train_target_2
        torch.randn(10, 28, 28),  # train_images_3
        torch.randint(0, 10, (10,)),  # train_target_3
        torch.randn(10, 28, 28),  # train_images_4
        torch.randint(0, 10, (10,)),  # train_target_4
        torch.randn(10, 28, 28),  # train_images_5
        torch.randint(0, 10, (10,)),  # train_target_5
    ]

    dataset = MyDataset("data/raw")

    # Test that it's a Dataset instance
    assert isinstance(dataset, Dataset)

    # Test that it has the correct length
    assert len(dataset) == 60  # 6 files * 10 samples each

    # Test that we can get an item
    image, target = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, torch.Tensor)


@patch('torch.load')
def test_corrupt_mnist(mock_load):
    """Test the corrupt_mnist function."""
    # Mock the data loading
    mock_load.side_effect = [
        torch.randn(100, 1, 28, 28),  # train_images
        torch.randint(0, 10, (100,)),  # train_target
        torch.randn(20, 1, 28, 28),   # test_images
        torch.randint(0, 10, (20,)),  # test_target
    ]

    train_set, test_set = corrupt_mnist()

    # Test that both are Dataset instances
    assert isinstance(train_set, Dataset)
    assert isinstance(test_set, Dataset)

    # Test lengths
    assert len(train_set) == 100
    assert len(test_set) == 20
