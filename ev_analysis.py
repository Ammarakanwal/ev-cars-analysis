# ============================================================
#  EV Population Data — Full Analysis
#  Dataset: Washington State Electric Vehicle Population Data
# ============================================================

#%% ── 0. Imports & Setup ────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.rcParams.update({
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'DejaVu Sans',
})

BLUE    = '#378ADD'
GREEN   = '#1D9E75'
CORAL   = '#D85A30'
PINK    = '#D4537E'
PURPLE  = '#7F77DD'
AMBER   = '#BA7517'
PALETTE = [BLUE, GREEN, CORAL, PINK, PURPLE, AMBER, '#0F6E56', '#639922', '#D4537E', '#888780']


#%% ── 1. Load Data ──────────────────────────────────────────
df = pd.read_csv(r'D:\iec\portfolio2\66634-ev-data\Electric_Vehicle_Population_Data.csv')

print(f"Shape       : {df.shape}")
print(f"Columns     : {list(df.columns)}")
df.info()


#%% ── 2. Null Check (tidy summary) ─────────────────────────
null_summary = df.isnull().sum().reset_index()
null_summary.columns = ['Column', 'Null Count']
null_summary['Null %'] = (null_summary['Null Count'] / len(df) * 100).round(2)
null_summary = null_summary[null_summary['Null Count'] > 0].sort_values('Null Count', ascending=False)
print("\nColumns with missing values:")
print(null_summary.to_string(index=False))


#%% ── 3. KPI Summary ────────────────────────────────────────
total       = len(df)
bev_count   = (df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)').sum()
phev_count  = (df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)').sum()
top_make    = df['Make'].value_counts().idxmax()
top_county  = df['County'].value_counts().idxmax()

print(f"\n{'='*40}")
print(f"  Total EVs registered : {total:,}")
print(f"  BEV                  : {bev_count:,}  ({bev_count/total*100:.1f}%)")
print(f"  PHEV                 : {phev_count:,}  ({phev_count/total*100:.1f}%)")
print(f"  Top manufacturer     : {top_make}")
print(f"  Top county           : {top_county}")
print(f"{'='*40}\n")


#%% ── 4. BEV vs PHEV Bar Chart ──────────────────────────────
type_counts = df['Electric Vehicle Type'].value_counts()
labels      = ['BEV\n(Battery Electric)', 'PHEV\n(Plug-in Hybrid)']
values      = [
    type_counts.get('Battery Electric Vehicle (BEV)', 0),
    type_counts.get('Plug-in Hybrid Electric Vehicle (PHEV)', 0),
]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, values, color=[BLUE, GREEN], width=0.45, edgecolor='white', linewidth=1.2)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 800,
        f'{val:,}\n({val/total*100:.1f}%)',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

ax.set_title('BEV vs PHEV — Total Registered Vehicles')
ax.set_ylabel('Number of Vehicles')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax.set_ylim(0, max(values) * 1.18)
plt.tight_layout()
plt.savefig('bev_vs_phev.png', bbox_inches='tight')
plt.show()


#%% ── 5. Adoption Rate Over Model Years ─────────────────────
adoption = df['Model Year'].value_counts().sort_index()
adoption = adoption[adoption.index >= 2010]          # trim noisy early years

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(adoption.index, adoption.values, alpha=0.12, color=BLUE)
ax.plot(adoption.index, adoption.values, color=BLUE, linewidth=2.5, marker='o', markersize=5)

for year, count in adoption.items():
    if count > 5000 or year in [2018, 2022]:
        ax.annotate(
            f'{count:,}',
            (year, count),
            textcoords='offset points', xytext=(0, 9),
            ha='center', fontsize=8, color=BLUE
        )

ax.set_title('EV Adoption — Registrations by Model Year')
ax.set_xlabel('Model Year')
ax.set_ylabel('Number of Vehicles')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax.set_xticks(adoption.index)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('adoption_over_years.png', bbox_inches='tight')
plt.show()


#%% ── 5b. Adoption Growth Rate (YoY %) ──────────────────────
yoy = adoption.pct_change() * 100
yoy = yoy.dropna()

fig, ax = plt.subplots(figsize=(11, 4))
colors = [GREEN if v >= 0 else CORAL for v in yoy.values]
ax.bar(yoy.index, yoy.values, color=colors, edgecolor='white', linewidth=0.8)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_title('Year-over-Year Adoption Growth Rate (%)')
ax.set_xlabel('Model Year')
ax.set_ylabel('YoY Growth (%)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_xticks(yoy.index)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('adoption_yoy_growth.png', bbox_inches='tight')
plt.show()


#%% ── 6. Manufacturer Market Share ──────────────────────────
make_counts = df['Make'].value_counts()
top_n       = 9
top_makes   = make_counts.head(top_n)
others      = make_counts.iloc[top_n:].sum()
make_plot   = pd.concat([top_makes, pd.Series({'Others': others})])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# — Horizontal bar chart
bars = ax1.barh(make_plot.index[::-1], make_plot.values[::-1],
                color=PALETTE[:len(make_plot)][::-1], edgecolor='white')
for bar, val in zip(bars, make_plot.values[::-1]):
    ax1.text(val + 200, bar.get_y() + bar.get_height() / 2,
             f'{val:,}  ({val/total*100:.1f}%)',
             va='center', fontsize=8.5)
ax1.set_title('Market Share by Manufacturer')
ax1.set_xlabel('Number of Vehicles')
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax1.set_xlim(0, make_plot.max() * 1.28)

# — Pie / donut chart
wedge_colors = PALETTE[:len(make_plot)]
wedges, texts, autotexts = ax2.pie(
    make_plot.values,
    labels=make_plot.index,
    autopct=lambda p: f'{p:.1f}%' if p > 2.5 else '',
    colors=wedge_colors,
    startangle=140,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    pctdistance=0.78
)
for t in autotexts:
    t.set_fontsize(8)
    t.set_fontweight('bold')
centre = plt.Circle((0, 0), 0.55, color='white')
ax2.add_patch(centre)
ax2.text(0, 0, f'{total:,}\nVehicles', ha='center', va='center',
         fontsize=10, fontweight='bold', color='#444')
ax2.set_title('Market Share Donut')

plt.suptitle('EV Manufacturer Market Share — Washington State', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('market_share.png', bbox_inches='tight')
plt.show()


#%% ── 7. Tesla vs All Others Over Time ──────────────────────
df['Is Tesla'] = df['Make'] == 'TESLA'
yearly_split = df.groupby(['Model Year', 'Is Tesla']).size().unstack(fill_value=0)
yearly_split.columns = ['Other Makers', 'Tesla']
yearly_split = yearly_split[yearly_split.index >= 2010]

fig, ax = plt.subplots(figsize=(11, 5))
yearly_split.plot(kind='area', stacked=True, ax=ax,
                  color=[BLUE, '#cccccc'], alpha=0.85)
ax.set_title('Tesla vs Other Manufacturers — Registrations by Year')
ax.set_xlabel('Model Year')
ax.set_ylabel('Number of Vehicles')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax.legend(['Other Makers', 'Tesla'], loc='upper left')
plt.tight_layout()
plt.savefig('tesla_vs_others.png', bbox_inches='tight')
plt.show()


#%% ── 8. Electric Range — BEV Only ─────────────────────────
bev_df   = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)'].copy()
bev_range = bev_df[bev_df['Electric Range'] > 0]['Electric Range']  # exclude 0-range records

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(bev_range, bins=30, color=BLUE, edgecolor='white', linewidth=0.8)

mean_r   = bev_range.mean()
median_r = bev_range.median()
ax.axvline(mean_r,   color=CORAL,  linestyle='--', linewidth=1.8, label=f'Mean: {mean_r:.0f} mi')
ax.axvline(median_r, color=PURPLE, linestyle=':',  linewidth=1.8, label=f'Median: {median_r:.0f} mi')

ax.set_title('Electric Range Distribution — BEVs Only')
ax.set_xlabel('Electric Range (miles)')
ax.set_ylabel('Number of Vehicles')
ax.legend()
plt.tight_layout()
plt.savefig('electric_range_distribution.png', bbox_inches='tight')
plt.show()

print(f"\nBEV Range Stats (excluding 0-range records):")
print(bev_range.describe().apply(lambda x: f'{x:.1f}'))


#%% ── 8b. Range by Make (Top 6 BEV makers) ──────────────────
top_bev_makes = (
    bev_df[bev_df['Electric Range'] > 0]
    .groupby('Make')['Electric Range']
    .count()
    .nlargest(6)
    .index.tolist()
)
range_data = bev_df[(bev_df['Make'].isin(top_bev_makes)) & (bev_df['Electric Range'] > 0)]

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(
    data=range_data,
    x='Make', y='Electric Range',
    order=top_bev_makes,
    palette=PALETTE[:6],
    width=0.5,
    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4},
    ax=ax
)
ax.set_title('Electric Range by Manufacturer — Top 6 BEV Makers')
ax.set_xlabel('')
ax.set_ylabel('Range (miles)')
plt.tight_layout()
plt.savefig('range_by_make.png', bbox_inches='tight')
plt.show()


#%% ── 9. Top Counties by EV Count ───────────────────────────
county_counts = df['County'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(county_counts.index[::-1], county_counts.values[::-1],
               color=[BLUE if i == 0 else GREEN if i < 3 else '#B5D4F4'
                      for i in range(len(county_counts)-1, -1, -1)],
               edgecolor='white')
for bar, val in zip(bars, county_counts.values[::-1]):
    ax.text(val + 200, bar.get_y() + bar.get_height() / 2,
            f'{val:,}  ({val/total*100:.1f}%)',
            va='center', fontsize=8.5)
ax.set_title('Top 10 Counties by EV Registrations')
ax.set_xlabel('Number of Vehicles')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax.set_xlim(0, county_counts.max() * 1.25)
plt.tight_layout()
plt.savefig('top_counties.png', bbox_inches='tight')
plt.show()


#%% ── 10. Preferred Maker per County (Top 5 Counties) ────────
top5_counties = df['County'].value_counts().head(5).index.tolist()
county_make   = (
    df[df['County'].isin(top5_counties)]
    .groupby(['County', 'Make'])
    .size()
    .reset_index(name='Count')
)
top3_per_county = (
    county_make
    .sort_values('Count', ascending=False)
    .groupby('County')
    .head(3)
    .reset_index(drop=True)
)

fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=False)
for ax, county in zip(axes, top5_counties):
    data = top3_per_county[top3_per_county['County'] == county]
    ax.barh(data['Make'][::-1], data['Count'][::-1],
            color=PALETTE[:3][::-1], edgecolor='white')
    ax.set_title(county, fontsize=10, fontweight='bold')
    ax.set_xlabel('Count', fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax.tick_params(labelsize=8)

plt.suptitle('Top 3 EV Makers per County — Top 5 Counties', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('maker_per_county.png', bbox_inches='tight')
plt.show()


#%% ── 11. CAFV Eligibility Breakdown ────────────────────────
cafv_col  = 'Clean Alternative Fuel Vehicle (CAFV) Eligibility'
cafv_vals = df[cafv_col].value_counts()

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(cafv_vals.index, cafv_vals.values,
               color=[GREEN, AMBER, CORAL][:len(cafv_vals)], edgecolor='white')
for bar, val in zip(bars, cafv_vals.values):
    ax.text(val + 100, bar.get_y() + bar.get_height() / 2,
            f'{val:,}  ({val/total*100:.1f}%)', va='center', fontsize=9)
ax.set_title('CAFV (Clean Alternative Fuel Vehicle) Eligibility')
ax.set_xlabel('Number of Vehicles')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax.set_xlim(0, cafv_vals.max() * 1.3)
plt.tight_layout()
plt.savefig('cafv_eligibility.png', bbox_inches='tight')
plt.show()


#%% ── 12. Numeric Correlation Heatmap ───────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))    # upper triangle only
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='Blues', linewidths=0.5,
    annot_kws={'size': 9}, ax=ax
)
ax.set_title('Numeric Feature Correlations')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.show()


#%% ── 13. BEV Adoption Share Over Time ──────────────────────
yearly_type = (
    df[df['Model Year'] >= 2010]
    .groupby(['Model Year', 'Electric Vehicle Type'])
    .size()
    .unstack(fill_value=0)
)
yearly_type_pct = yearly_type.div(yearly_type.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(11, 5))
yearly_type_pct.plot(kind='bar', stacked=True, ax=ax,
                     color=[BLUE, GREEN], edgecolor='white', linewidth=0.5, width=0.75)
ax.set_title('BEV vs PHEV Share Over Time (%)')
ax.set_xlabel('Model Year')
ax.set_ylabel('Share (%)')
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.legend(['BEV', 'PHEV'], loc='upper left')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('bev_phev_share_over_time.png', bbox_inches='tight')
plt.show()


#%% ── 14. Base MSRP Analysis (non-zero records) ─────────────
msrp_df = df[df['Base MSRP'] > 0].copy()

if len(msrp_df) > 100:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Distribution
    axes[0].hist(msrp_df['Base MSRP'], bins=40, color=PURPLE, edgecolor='white', linewidth=0.7)
    axes[0].set_title('Base MSRP Distribution')
    axes[0].set_xlabel('Base MSRP ($)')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${int(x/1000)}k'))

    # MSRP vs Range
    axes[1].scatter(msrp_df['Electric Range'], msrp_df['Base MSRP'],
                    alpha=0.4, color=BLUE, s=15)
    axes[1].set_title('Electric Range vs Base MSRP')
    axes[1].set_xlabel('Electric Range (miles)')
    axes[1].set_ylabel('Base MSRP ($)')
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${int(x/1000)}k'))

    plt.tight_layout()
    plt.savefig('msrp_analysis.png', bbox_inches='tight')
    plt.show()
else:
    print("Not enough non-zero MSRP records to plot (common in this dataset — most MSRP values are 0).")


#%% ── 15. Full Summary Dashboard (1 figure, 6 panels) ───────
# ── recompute all needed variables so this cell runs standalone ──
total      = len(df)
bev_count  = (df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)').sum()
phev_count = (df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)').sum()

adoption   = df['Model Year'].value_counts().sort_index()
adoption   = adoption[adoption.index >= 2010]

bev_df     = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)'].copy()
bev_range  = bev_df[bev_df['Electric Range'] > 0]['Electric Range']

yearly_type = (
    df[df['Model Year'] >= 2010]
    .groupby(['Model Year', 'Electric Vehicle Type'])
    .size()
    .unstack(fill_value=0)
)
yearly_type_pct = yearly_type.div(yearly_type.sum(axis=1), axis=0) * 100
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Washington State EV Population — Full Dashboard', fontsize=15, fontweight='bold', y=1.01)

# Panel 1: BEV vs PHEV
ax = axes[0, 0]
ax.bar(['BEV', 'PHEV'], [bev_count, phev_count], color=[BLUE, GREEN], edgecolor='white', width=0.5)
ax.set_title('BEV vs PHEV')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

# Panel 2: Adoption over time
ax = axes[0, 1]
ax.fill_between(adoption.index, adoption.values, alpha=0.15, color=BLUE)
ax.plot(adoption.index, adoption.values, color=BLUE, linewidth=2, marker='o', markersize=3)
ax.set_title('Adoption by Model Year')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax.tick_params(axis='x', rotation=45)

# Panel 3: Top 8 makes
ax = axes[0, 2]
top8 = df['Make'].value_counts().head(8)
ax.barh(top8.index[::-1], top8.values[::-1], color=PALETTE[:8][::-1], edgecolor='white')
ax.set_title('Top 8 Manufacturers')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

# Panel 4: Electric range histogram (BEV > 0)
ax = axes[1, 0]
ax.hist(bev_range, bins=25, color=BLUE, edgecolor='white', linewidth=0.7)
ax.axvline(bev_range.mean(), color=CORAL, linestyle='--', linewidth=1.5, label=f'Mean {bev_range.mean():.0f}mi')
ax.set_title('BEV Electric Range')
ax.set_xlabel('Miles')
ax.legend(fontsize=8)

# Panel 5: Top 8 counties
ax = axes[1, 1]
top8_county = df['County'].value_counts().head(8)
ax.barh(top8_county.index[::-1], top8_county.values[::-1],
        color=[BLUE if i == 7 else '#B5D4F4' for i in range(8)], edgecolor='white')
ax.set_title('Top 8 Counties')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

# Panel 6: BEV vs PHEV share over time
ax = axes[1, 2]
yearly_type_pct.plot(kind='area', stacked=True, ax=ax,
                     color=[BLUE, GREEN], alpha=0.8, legend=False)
ax.set_title('BEV/PHEV Share Over Time')
ax.set_xlabel('Model Year')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('full_dashboard.png', bbox_inches='tight', dpi=150)
plt.show()

print("\nAll charts saved as PNG files in the working directory.")
print("Done!")
