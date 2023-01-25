import pandas as pd
from pathlib import Path

data_path = Path('test.xlsx')
data = pd.read_excel(data_path, None)
print(data)