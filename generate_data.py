import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
random.seed(42)
np.random.seed(42)

records = []

for i in range(50):
    is_fraud = 1 if random.random() < 0.1 else 0  # ~10% fraud (about 5 out of 50)

    record = {
        "transaction_id": fake.uuid4(),
        "amount": round(random.uniform(500, 5000), 2) if is_fraud else round(random.uniform(5, 500), 2),
        "hour": random.randint(0, 3) if is_fraud else random.randint(8, 20),
        "merchant_category": random.choice(["electronics", "jewelry", "crypto"]) if is_fraud else random.choice(["grocery", "restaurant", "clothing", "pharmacy"]),
        "is_foreign": 1 if is_fraud and random.random() < 0.8 else 0,
        "customer_age": random.randint(18, 40) if is_fraud else random.randint(25, 70),
        "is_fraud": is_fraud
    }
    records.append(record)

df = pd.DataFrame(records)
df.to_csv("transactions.csv", index=False)

print(f"Dataset created: {len(df)} transactions")
print(f"Fraud cases: {df['is_fraud'].sum()}")
print(f"Legit cases: {(df['is_fraud'] == 0).sum()}")
print(df.head())
