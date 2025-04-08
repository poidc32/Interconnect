# %% Distribucion de variables numericas -----------------------------------

num_cols = ['monthly_charges', 'total_charges']

for col in num_cols:
    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    sns.histplot(df_full[col], kde=True, bins=30)
    plt.title(f'Distribución de {col}')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_full[col])
    plt.title(f'Boxplot de {col}')

    plt.tight_layout()
    plt.show()

# %% Comparación de distribución según Churn ---------------------------------

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='churn', y=col, data=df_full)
    plt.title(f'{col} vs Churn')
    plt.xlabel('Churn')
    plt.ylabel(col)
    plt.show()

# %% Análisis de variables categóricas vs Churn ------------------------------

cat_cols = [
    'gender', 'partner', 'dependents', 'contract_type',
    'paperless_billing', 'payment_method', 'internet_service',
    'online_security', 'online_backup', 'device_protection',
    'tech_support', 'streaming_tv', 'streaming_movies',
    'multiple_lines'
]

n = len(cat_cols)

cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))

axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(data=df_full, x=col, hue='churn', ax=axes[i])
    axes[i].set_title(f'{col} vs Churn')
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %% roporción de churn para cada categoría -----------------------------------

cols = 3
rows = math.ceil(len(cat_cols) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    churn_rate = df_full.groupby(col)['churn'].mean().sort_values(ascending=True)
    sns.barplot(x=churn_rate.values, y=churn_rate.index, ax=axes[i])
    axes[i].set_title(f'Tasa de Churn por {col}')
    axes[i].set_xlabel('Proporción de Churn')
    axes[i].set_ylabel(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
