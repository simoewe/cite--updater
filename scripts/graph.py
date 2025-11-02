import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import datetime
import pandas as pd
import re

# Set up LaTeX-style plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.unicode_minus'] = False

# Regular expressions from the notebook
paper_info = re.compile(r'^(?:arXiv:)?(\d+\.\d+)v(\d+)(?:.pdf[:-]| )')

# Read and process dates
dates = {}
with open('danica/dates', 'r') as f:
    for line in f:
        arxiv, ver = paper_info.match(line.strip()).groups()
        ver = int(ver)
        dates[arxiv, ver] = pd.to_datetime(datetime.datetime.strptime(line[line.rindex(']') + 1:].strip(), '%d %b %Y').date())

# Process citation data
right = re.compile(r'(Danica (\s+ J\.?)? \s+ Sutherland) | Sutherland, \s+ Danica', re.IGNORECASE | re.VERBOSE)
wrong = re.compile(r'(Dougal (\s+ J\.?)? \s+ Sutherland) | Sutherland, \s+ Dougal', re.IGNORECASE | re.VERBOSE)
inits = re.compile(r'(\bD\.?\s*(J\.?)?\s+Sutherland)|(Sutherland,\s+D\b\.?\s*(J\.?)?)', re.IGNORECASE)

# Create DataFrame with citation information
records = []
with open('danica/greps', 'r') as f:
    for line in f:
        if line == '--\n':
            continue
        try:
            arxiv, ver = paper_info.match(line).groups()
            ver = int(ver)
            date = dates.get((arxiv, ver))
            if date:
                records.append({
                    'date': date,
                    'arxiv': arxiv,
                    'version': ver,
                    'Danica': bool(right.search(line)),
                    'D[eadname]': bool(wrong.search(line)),
                    'D.J.': bool(inits.search(line))
                })
        except (AttributeError, KeyError):
            continue

# Create DataFrame and process as in notebook
all_info = pd.DataFrame.from_records(records)
all_info.set_index('date', inplace=True)
all_info.sort_index(inplace=True)

# Define the exact 12-month period
start_date = pd.Timestamp('2020-12-01')
end_date = pd.Timestamp('2021-12-01')

# Filter and process data for the specific period
citation_data = all_info[all_info.version == 1] \
    .drop(columns=['version', 'arxiv']).rolling('90d').sum().div(3)

# Slice to exact date range
citation_data = citation_data[start_date:end_date]

# Analysis points
proceedings_date = pd.Timestamp('2021-01-14')
scholar_date = pd.Timestamp('2021-04-14')

# Get values at key points
one_month = pd.Timedelta(days=30)
pre_proceedings = citation_data.loc[proceedings_date - one_month:proceedings_date].mean()
post_proceedings = citation_data.loc[proceedings_date:proceedings_date + one_month].mean()

# For Scholar impact, look at months 6-8 (peak period from graph)
peak_period = citation_data.loc['2021-06-01':'2021-08-31'].mean()

# For final state, look at last 2 months
final_period = citation_data.loc['2021-10-01':'2021-11-30'].mean()

# Calculate percentages
total_post_proceedings = post_proceedings.sum()
total_peak = peak_period.sum()
total_final = final_period.sum()

correct_pct_proceedings = (post_proceedings['Danica'] / total_post_proceedings) * 100
peak_correct_pct = (peak_period['Danica'] / total_peak) * 100
deadname_pct_final = (final_period['D[eadname]'] / total_final) * 100

# Print analysis
print(f"\nAnalysis Results:")
print(f"After proceedings update: {correct_pct_proceedings:.1f}% correct name usage")
print(f"Peak period (Jun-Aug): {peak_correct_pct:.1f}% correct name usage")
print(f"Final period deadname usage (Oct-Nov): {deadname_pct_final:.1f}%")
print(f"Months between updates: {(scholar_date - proceedings_date).days / 30:.1f}")

# Create the plot
plt.figure(figsize=(8, 4))

# Create evenly spaced month points
month_points = pd.date_range(start_date, end_date, periods=12)

# Plot lines
plt.plot(citation_data.index, citation_data['Danica'], color='teal', label='Correct name citations', linewidth=1.5)
plt.plot(citation_data.index, citation_data['D[eadname]'], 'r-', label='Deadname citations', linewidth=1.5)
plt.plot(citation_data.index, citation_data['D.J.'], 'gray', label='Initial-only citations', linewidth=1.5)

# Add vertical lines for key events
plt.axvline(pd.Timestamp('2021-01-05'), color='k', linestyle=':', linewidth=1)
plt.axvline(pd.Timestamp('2021-04-14'), color='k', linestyle=':', linewidth=1)

# Customize the plot with bold axis labels only
plt.xlabel('Time (months)', fontsize=18, fontweight='bold')
plt.ylabel('Smoothed citations', fontsize=18, fontweight='bold')

# Set axis limits
plt.ylim(0, 15)
plt.xlim(start_date, end_date)

# Set x-axis ticks to show months 0-11
plt.xticks(month_points, range(12), fontsize=12)
plt.yticks(range(0, 16, 5), fontsize=12)

# Add annotations
plt.annotate('Proceedings\nupdated with name', 
            xy=(pd.Timestamp('2021-01-05'), 14),
            xytext=(5, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=12)

plt.annotate('Search engine\nsync', 
            xy=(pd.Timestamp('2021-04-14'), 14),
            xytext=(5, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=12)

# Add legend in top right
plt.legend(loc='upper right',
          fontsize=10,
          frameon=True,
          framealpha=0.9)

# Add grid
plt.grid(True, linestyle=':', alpha=0.2)

# Adjust layout with minimal margins
plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.15)

# Save the figure
plt.savefig('cite-counts.pdf', dpi=300, bbox_inches='tight')
plt.close()