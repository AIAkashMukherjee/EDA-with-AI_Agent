import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tempfile,traceback

def save_fig(fig):
    f=tempfile.NamedTemporaryFile(delete=False,suffix='.png')
    fig.savefig(f.name,bbox_inches='tight',dpi=100)
    plt.close(fig)
    return f.name

def generate_graphs(df:pd.DataFrame):
    visualizations = []
    saved_files = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [col for col in df.select_dtypes(include='object') if 1 < df[col].nunique() < 30]

    try:
        # 1. Histograms for numeric columns
        if numeric_cols:
            fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(8, 4*len(numeric_cols)))
            if len(numeric_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                sns.histplot(df[col], kde=True, ax=ax) # type: ignore
                ax.set_title(f'Histogram of {col}')
            plt.tight_layout()
            hist_path = save_fig(fig)
            saved_files.append(hist_path)
            visualizations.append(('Histograms of Numeric Columns', hist_path))

        # 2. Boxplots for numeric columns
        if numeric_cols:
            fig, axes = plt.subplots(nrows=1, ncols=len(numeric_cols), figsize=(4*len(numeric_cols), 6))
            if len(numeric_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                sns.boxplot(y=df[col], ax=ax)
                ax.set_title(f'Boxplot of {col}')
            plt.tight_layout()
            box_path = save_fig(fig)
            saved_files.append(box_path)
            visualizations.append(('Boxplots of Numeric Columns', box_path))

        # 3. Countplots for categorical columns
        if cat_cols:
            fig, axes = plt.subplots(nrows=len(cat_cols), ncols=1, figsize=(8, 4*len(cat_cols)))
            if len(cat_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, cat_cols):
                sns.countplot(x=df[col], ax=ax)
                ax.set_title(f'Countplot of {col}')
                ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            count_path = save_fig(fig)
            saved_files.append(count_path)
            visualizations.append(('Countplots of Categorical Columns', count_path))

        # 4. Pairplot (if reasonable number of numeric columns)
        if len(numeric_cols) > 1 and len(numeric_cols) <= 5:
            pairplot = sns.pairplot(df[numeric_cols])
            pair_path = save_fig(pairplot.fig)
            saved_files.append(pair_path)
            visualizations.append(('Pairplot of Numeric Columns', pair_path))

        # 5. Violin plots (numeric vs categorical if available)
        if numeric_cols and cat_cols:
            # Use first numeric and first categorical column for example
            num_col = numeric_cols[0]
            cat_col = cat_cols[0]
            fig = plt.figure(figsize=(10, 6))
            sns.violinplot(x=df[cat_col], y=df[num_col])
            plt.title(f'Violin Plot of {num_col} by {cat_col}')
            plt.xticks(rotation=45)
            violin_path = save_fig(fig)
            saved_files.append(violin_path)
            visualizations.append((f'Violin Plot of {num_col} by {cat_col}', violin_path))

        # 6. Correlation heatmap
        if len(numeric_cols) > 1:
            fig = plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            heatmap_path = save_fig(fig)
            saved_files.append(heatmap_path)
            visualizations.append(('Correlation Heatmap', heatmap_path))

    except Exception as e:
        print(f"Error generating graphs: {e}")
        traceback.print_exc()

    return visualizations, saved_files