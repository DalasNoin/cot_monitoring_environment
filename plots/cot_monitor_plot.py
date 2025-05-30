import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick # Added for PercentFormatter
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate CoT Monitor plots')
parser.add_argument('--pdf', action='store_true', help='Save plots as PDF instead of PNG')
args = parser.parse_args()

# Set output extension based on flag
output_ext = '.pdf' if args.pdf else '.png'

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.3)

# Load the CSV data
df = pd.read_csv('plots/cot_monitor.csv')

# Extract model name without organization prefix for cleaner display
df['Model_Short'] = df['Model'].apply(lambda x: x.split('/')[-1])

# Add model size to display name for better readability
df['Model_Display'] = df['Model_Short'] + ' (' + df['Params (in b)'].astype(str) + 'B)'

# Sort models by parameter size
model_param_dict = df.groupby('Model_Display')['Params (in b)'].first().to_dict()
model_order = sorted(model_param_dict.keys(), key=lambda x: model_param_dict[x])

# Sort overseer names in a meaningful order
overseer_order = [
    'claude-3.5-sonnet',
    'gpt-4o',
    'gpt-4o-mini',
    'gemini-2.0-flash',
    'llama-3.3-70b-instruct',
    'qwq-32b'
]

# Ensure all overseer categories exist in our data
valid_overseers = [o for o in overseer_order if o in df['Overseer'].unique()]

# Create categorical type with our desired order
df['Overseer_Cat'] = pd.Categorical(df['Overseer'], categories=valid_overseers, ordered=True)

# 1. Create a heatmap of jailbreak success rates
plt.figure(figsize=(14, 10))
pivot_df = df.pivot_table(index='Model_Display', columns='Overseer', values='Score with jailbreak')

# Reorder rows and columns
pivot_df = pivot_df.loc[model_order]
pivot_df = pivot_df[valid_overseers]

# Create the heatmap
ax = sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd_r', linewidths=0.5, annot_kws={"size": 14})
plt.title('CoT Monitor: Jailbreak Success Rate by Model and Overseer', fontsize=20)
plt.xlabel('Overseer Model', fontsize=16)
plt.ylabel('Target Model', fontsize=16)

# Add parameter size as text to the left of the model names
for i, model in enumerate(pivot_df.index):
    param_size = model_param_dict[model]
    plt.text(-0.5, i + 0.5, f"{param_size}B", ha='right', va='center', fontsize=12, 
             fontweight='bold', color='darkblue')

plt.tight_layout()
plt.savefig(f'plots/cot_monitor_heatmap{output_ext}', dpi=300)

# 2. Create grouped bar chart by model and overseer
plt.figure(figsize=(16, 10))
ax = sns.barplot(
    x='Model_Display', 
    y='Score with jailbreak', 
    hue='Overseer',
    data=df,
    order=model_order,
    hue_order=valid_overseers,
    palette='viridis'
)

plt.title('CoT Monitor: Jailbreak Success Rate by Target Model and Overseer', fontsize=20)
plt.xlabel('Target Model', fontsize=16)
plt.ylabel('Jailbreak Success Rate', fontsize=16)
plt.ylim(0, df['Score with jailbreak'].max() * 1.1)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Overseer Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(f'plots/cot_monitor_barplot{output_ext}', dpi=300)

# 3. Boxplot showing the distribution of jailbreak success rates by overseer
plt.figure(figsize=(14, 8))
ax = sns.boxplot(
    x='Overseer', 
    y='Score with jailbreak', 
    data=df,
    order=valid_overseers,
    palette='Set3'
)

# Add individual points
sns.stripplot(
    x='Overseer', 
    y='Score with jailbreak', 
    data=df,
    order=valid_overseers,
    color='black',
    size=4,
    alpha=0.6
)

plt.title('CoT Monitor: Distribution of Jailbreak Success Rates by Overseer', fontsize=20)
plt.xlabel('Overseer Model', fontsize=16)
plt.ylabel('Jailbreak Success Rate', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'plots/cot_monitor_overseer_boxplot{output_ext}', dpi=300)

# 4. Boxplot showing distribution by model parameter size
plt.figure(figsize=(14, 8))
# Create parameter size categories
param_bins = [0, 10, 50, 100, 1000]
param_labels = ['<10B', '10-50B', '50-100B', '>100B']
df['Param_Category'] = pd.cut(df['Params (in b)'], bins=param_bins, labels=param_labels)

ax = sns.boxplot(
    x='Param_Category', 
    y='Score with jailbreak', 
    data=df,
    palette='Blues'
)

# Add individual points
sns.stripplot(
    x='Param_Category', 
    y='Score with jailbreak', 
    data=df,
    color='black',
    size=4,
    alpha=0.6
)

plt.title('CoT Monitor: Jailbreak Success Rates by Model Size', fontsize=20)
plt.xlabel('Model Parameter Size', fontsize=16)
plt.ylabel('Jailbreak Success Rate', fontsize=16)
plt.tight_layout()
plt.savefig(f'plots/cot_monitor_param_boxplot{output_ext}', dpi=300)

# 5. Calculate and display average success rate by overseer
overseer_means = df.groupby('Overseer')['Score with jailbreak'].mean().sort_values().reset_index()
overseer_means = overseer_means.sort_values('Score with jailbreak')  # Sort by success rate

plt.figure(figsize=(14, 10)) # Adjusted figure size for larger fonts
bars = sns.barplot(
    x='Overseer', 
    y='Score with jailbreak', 
    data=overseer_means,
    order=overseer_means['Overseer'],
    palette='viridis_r'
)

plt.title('Average Jailbreak Success Rate by Overseer Model', fontsize=34, color='black')
plt.xlabel('Overseer Model', fontsize=30, color='black')
plt.ylabel('Average Jailbreak Success Rate', fontsize=30, color='black')
plt.ylim(0, 1) # Y-axis from 0 to 1 for percentage formatting
plt.xticks(rotation=45, ha='right', fontsize=28, color='black')
plt.yticks(fontsize=28, color='black')

# Format y-axis as percentage
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# Add value labels on top of bars, formatted as percentages
for i, bar in enumerate(bars.patches):
    bars.text(bar.get_x() + bar.get_width()/2, 
              bar.get_height() + 0.01, 
              f'{bar.get_height()*100:.1f}%', # Convert to percentage and format
              ha='center', fontsize=26, color='black', fontweight='bold')

plt.tight_layout()
plt.savefig(f'plots/cot_monitor_overseer_performance{output_ext}', dpi=300, transparent=True)

print(f"CoT Monitor plots saved in the plots directory as {output_ext[1:]} files") 