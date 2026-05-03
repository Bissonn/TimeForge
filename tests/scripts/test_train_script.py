import pytest
from unittest.mock import patch


# --- CLI Argument Parsing Tests ---

def test_cli_mutually_exclusive_modes(mocker):
    """Verify that argparse raises an error if mutually exclusive modes are combined."""
    from scripts.train import parse_arguments

    # Mock sys.argv to simulate command-line usage
    mock_argv = mocker.patch('sys.argv', [
        'scripts/train.py',
        '--experiment', 'test',
        '--config-path', 'dummy.yaml',
        '--optimize',  # Exclusive arg 1
        '--evaluate'   # Exclusive arg 2
    ])

    with pytest.raises(SystemExit):
        parse_arguments()

