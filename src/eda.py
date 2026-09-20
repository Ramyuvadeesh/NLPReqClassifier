from load_data import load_dataset

df = load_dataset()

print("Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nClass Distribution:")
print(df["class"].value_counts())

print("\nPercentage Distribution:")
print(df["class"].value_counts(normalize=True) * 100)

df["Text_Length"] = df["RequirementText"].apply(len)

print("\nText Length Statistics:")
print(df["Text_Length"].describe())

print("\nSample Requirements:")
print(df["RequirementText"].sample(10))

print(df["ProjectID"].nunique())

print(df["ProjectID"].value_counts())