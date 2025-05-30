import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate refusal rate plots')
parser.add_argument('--pdf', action='store_true', help='Save plots as PDF instead of PNG')
args = parser.parse_args()

# Set output extension based on flag
output_ext = '.pdf' if args.pdf else '.png'

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.3)  # Increased font scale

# Load the CSV data
file_path = 'plots/refusal rates - DNA dataset - results.csv'
data = []

# Attacker display names mapping
attacker_display_names = {
    'Baseline': 'Baseline',
    'universal_jailbreak': 'Universal Jailbreak',
    'DeepSeek-R1-V1': 'DeepSeek-R1-V1',
    'DeepSeek-R1-V2': 'DeepSeek-R1-V2'
}

# Define the order of attackers
attacker_order = ['Baseline', 'universal_jailbreak', 'DeepSeek-R1-V1', 'DeepSeek-R1-V2']

# Parse the CSV maintaining its structure
current_attacker = None
with open(file_path, 'r') as file:
    lines = file.readlines()
    headers = lines[0].strip().split(',')
    
    for line in lines[1:]:
        values = line.strip().split(',')
        if values[0] == 'attacker':
            current_attacker = values[1]
        elif values[0] and current_attacker:  # This is a model row
            model_name = values[0]
            attack_success_rate = float(values[-1]) if values[-1] else 0
            data.append({
                'Attacker': current_attacker,
                'Attacker_Display': attacker_display_names.get(current_attacker, current_attacker),
                'Model': model_name,
                'Success Rate': attack_success_rate
            })

# Create DataFrame
result_df = pd.DataFrame(data)

# Create a categorical type with our desired order
attacker_cat_type = pd.CategoricalDtype(categories=attacker_order, ordered=True)
result_df['Attacker_Ordered'] = result_df['Attacker'].astype(attacker_cat_type)

# Sort the dataframe by the ordered attacker column
result_df = result_df.sort_values('Attacker_Ordered')

# Calculate mean success rate by attacker
attacker_means = result_df.groupby(['Attacker', 'Attacker_Display'])['Success Rate'].mean().reset_index()
# Ensure attacker means are also ordered
attacker_means['Attacker_Ordered'] = attacker_means['Attacker'].astype(attacker_cat_type)
attacker_means = attacker_means.sort_values('Attacker_Ordered')

print("Mean Attack Success Rate by Attacker:")
print(attacker_means[['Attacker_Display', 'Success Rate']])

# Plot 1: Bar chart of mean attack success rates
plt.figure(figsize=(12, 7))
bars = sns.barplot(x='Attacker_Display', y='Success Rate', data=attacker_means, 
                  palette='viridis', order=attacker_means['Attacker_Display'])
plt.title('Mean Attack Success Rate by Model-Written Jailbreak Type', fontsize=20)
plt.xlabel('Attacker Type', fontsize=16)
plt.ylabel('Success Rate (%)', fontsize=16)
plt.ylim(0, 100)

# Add value labels on top of bars
for i, bar in enumerate(bars.patches):
    bars.text(bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 1, 
            f'{bar.get_height():.1f}%', 
            ha='center', fontsize=14)

plt.tight_layout()
plt.savefig(f'plots/mean_attack_success{output_ext}', dpi=300)

# Plot 2: Heatmap of all results
plt.figure(figsize=(14, 9))
# Create pivot table with display names
pivot_df = result_df.pivot(index='Model', columns='Attacker_Display', values='Success Rate')
# Reorder columns
pivot_df = pivot_df[[attacker_display_names[a] for a in attacker_order]]
sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5, annot_kws={"size": 14})
plt.title('Model-Written Jailbreak Success Rate by Model and Attacker Type', fontsize=20)
plt.xlabel('Attacker Type', fontsize=16)
plt.ylabel('Model', fontsize=16)
plt.tight_layout()
plt.savefig(f'plots/attack_success_heatmap{output_ext}', dpi=300)

# Plot 3: Success rate by model and attacker (grouped bar chart)
plt.figure(figsize=(16, 9))
# Define a consistent order for models too
model_order = result_df['Model'].unique()
sns.barplot(x='Model', y='Success Rate', hue='Attacker_Display', data=result_df, 
           hue_order=[attacker_display_names[a] for a in attacker_order],
           order=model_order)
plt.title('Model Vulnerability to Model-Written Jailbreak Attacks', fontsize=20)
plt.xlabel('Model', fontsize=16)
plt.ylabel('Attack Success Rate (%)', fontsize=16)
plt.ylim(0, max(result_df['Success Rate'])*1.1)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attacker Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(f'plots/model_vulnerability{output_ext}', dpi=300)

# Plot 4: NEW - Success rate by attacker type with models in legend
# Modified for larger fonts, transparent BG, no legend, square shape, specific color mapping
fig4, ax4 = plt.subplots(figsize=(12, 12)) # Square shape

# Calculate model means across attackers (used for ordering hues)
model_means_for_order = result_df.groupby('Model')['Success Rate'].mean().reset_index()
model_means_for_order = model_means_for_order.sort_values('Success Rate', ascending=False)
# This is the order models will appear if multiple hues are present for an attacker
hue_model_order = model_means_for_order['Model'].tolist()

# Define specific models and their desired colors
claude_model_name = 'anthropic/claude-3.7-sonnet'
gpt41_model_name = 'openai/gpt-4.1'

std_tab10_colors = sns.color_palette('tab10', 10) # Ensure we have enough base colors
purple_color = std_tab10_colors[4] # Standard tab10 purple
green_color = std_tab10_colors[2]  # Standard tab10 green

# Build custom palette
custom_palette = {}
other_color_idx = 0
# Use a copy of tab10 colors that we can pop from, excluding purple and green initially
available_other_colors = [std_tab10_colors[i] for i in [0, 1, 3, 5, 6, 7, 8, 9]]

# Assign colors based on the hue_model_order
for model_name in hue_model_order:
    if model_name == claude_model_name:
        custom_palette[model_name] = purple_color
    elif model_name == gpt41_model_name:
        custom_palette[model_name] = green_color
    else:
        if available_other_colors:
            custom_palette[model_name] = available_other_colors.pop(0)
        else: # Fallback if more models than unique other colors
            custom_palette[model_name] = std_tab10_colors[(other_color_idx + 6) % 10] # Cycle through tab10, avoiding clash
            other_color_idx +=1


# Barplot with attacker on x-axis and model as hue
sns.barplot(
    x='Attacker_Display', 
    y='Success Rate', 
    hue='Model', 
    data=result_df,
    order=[attacker_display_names[a] for a in attacker_order],
    hue_order=hue_model_order, # Order models consistently
    palette=custom_palette,
    ax=ax4
)

# Add a text annotation showing the mean for each attacker
for i, attacker_display_name in enumerate([attacker_display_names[a] for a in attacker_order]):
    mean_value = attacker_means[attacker_means['Attacker_Display'] == attacker_display_name]['Success Rate'].values[0]
    ax4.text(i, 5, f"Mean: {mean_value:.1f}%", 
             ha='center', va='bottom', fontsize=26, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.4'), color='black')

ax4.set_title('Attacker Success Rate by Jailbreak Type', fontsize=34, color='black')
ax4.set_xlabel('Attacker Type', fontsize=30, color='black')
ax4.set_ylabel('Attack Success Rate (%)', fontsize=30, color='black')
ax4.set_ylim(0, 100)
ax4.tick_params(axis='x', labelsize=28, rotation=45, colors='black')
ax4.tick_params(axis='y', labelsize=28, colors='black')

# Remove legend
ax4.get_legend().remove()

# Add text annotation for color key
fig4.text(0.02, 0.02, "Purple: Claude-3.7-Sonnet, Green: GPT-4.1", fontsize=24, color='black', ha='left')


plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for figtext
plt.savefig(f'plots/attacker_performance{output_ext}', dpi=300, transparent=True)

print(f"Plots saved in the plots directory as {output_ext[1:]} files")