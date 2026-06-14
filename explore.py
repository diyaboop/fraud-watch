import pandas as pd

#Loading files

transaction = pd.read_csv('data/train_transaction.csv')
identity = pd.read_csv('data/train_identity.csv')

print(f"Transaction shape: {transaction.shape}")
print(f"Identity shape: {identity.shape}")
print(f"\nTransaction columns: {transaction.columns}")
print(f"\nIdentity columns: {identity.columns}")
print(f"\nFraud rate: {transaction['isFraud'].mean()}")
print(f"\nMissing values in transaction data: {transaction.isnull().sum().sort_values(ascending=False)}")
