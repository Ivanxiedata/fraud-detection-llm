import sys
import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_data import DataPreparer

@pytest.fixture
def preparer():
    return DataPreparer()

def test_format_transaction(preparer):
    row = {
        'Time': 100,
        'Amount': 50.0,
        'V1': 1.0, 'V2': 2.0, 'V3': 3.0, 'V4': 4.0, 'V5': 5.0,
        'V6': 6.0, 'V7': 7.0, 'V8': 8.0, 'V9': 9.0, 'V10': 10.0,
        'V11': 11.0, 'V12': 12.0, 'V13': 13.0, 'V14': 14.0, 'V15': 15.0,
        'V16': 16.0, 'V17': 17.0, 'V18': 18.0, 'V19': 19.0, 'V20': 20.0,
        'V21': 21.0, 'V22': 22.0, 'V23': 23.0, 'V24': 24.0, 'V25': 25.0,
        'V26': 26.0, 'V27': 27.0, 'V28': 28.0
    }
    description = preparer.format_transaction(row)
    assert "Time: 100" in description
    assert "Amount: 50.00" in description
    assert "V1:1.00" in description
    assert "V28:28.00" in description

def test_balance_data(preparer):
    # Create synthetic dataframe
    data = {
        'Class': [1] * 10 + [0] * 100,  # 10 fraud, 100 normal
        'Time': range(110),
        'Amount': range(110)
    }
    # Add V1-V28 columns
    for i in range(1, 29):
        data[f'V{i}'] = range(110)
        
    df = pd.DataFrame(data)
    
    balanced_df, df_fraud, df_normal = preparer.balance_data(df)
    
    assert len(df_fraud) == 10
    assert len(df_normal) == 100
    # Balanced df should have all 10 frauds + 20 normals (2x fraud) = 30 total
    assert len(balanced_df) == 30
    assert len(balanced_df[balanced_df['Class'] == 1]) == 10
    assert len(balanced_df[balanced_df['Class'] == 0]) == 20
