# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "features.csv"

try:
    features_df = pd.read_csv(CSV_PATH)

    print("DataFrame loaded successfully!")
    # Compute the correlation matrix for all numerical features in the DataFrame.
    print("\\n--- Computing Correlation Matrix ---")
    
    # The .corr() method calculates the pairwise correlation of columns, excluding NA/null values.
    # It returns a new DataFrame representing the matrix.
    correlation_matrix = features_df.corr()
    
    print("Correlation matrix computed successfully.")
    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Visualize the correlation matrix as a heatmap
    print("\n--- Generating Heatmap of Feature Correlations ---")
    
    # Set up the matplotlib figure
    # A larger figure size is needed to make the heatmap readable.
    plt.figure(figsize=(18, 15))

    # Create the heatmap using Seaborn
    # cmap='coolwarm': This is a "diverging" colormap, ideal for correlation matrices.
    #   Positive correlations will be warm (red), negative will be cool (blue),
    #   and correlations near zero will be neutral.
    # annot=False: For a large matrix like this, annotating each cell with its
    #   value would make the plot unreadable. We leave it false.
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)

    # Set the plot title
    plt.title('Correlation Matrix of Music Features', fontsize=20)

    # Display the plot
    plt.tight_layout()
    plt.show()
    # Displaying the top of the correlation matrix to get a feel for it.
    # A full 29x29 matrix is too large to display effectively in the console.
    print("Top 5 rows of the correlation matrix:")
    print(correlation_matrix.head())
    # --- Data Info ---
    print("\n--- DataFrame Info ---")
    features_df.info()

    # --- Statistical Summary ---
    print("\n--- DataFrame Statistical Summary ---")
    print(features_df.describe())

    # --- Missing Values ---
    print("\n--- Missing Values Check ---")
    missing_values_count = features_df.isnull().sum()
    print("Number of missing values per column:")
    print(missing_values_count)

    if missing_values_count.sum() == 0:
        print("\nConclusion: The dataset has no missing values. No handling is required.")
    else:
        print("\nConclusion: The dataset contains missing values. Handling is required.")

    # --- Preview Data ---
    print("\nFirst 5 rows of the dataset:")
    print(features_df.head())

    # =====================================================
    # --- VISUALIZATION SECTION ---
    # =====================================================

    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]

    sns.set_style("whitegrid")

    # -------------------------------
    # 1. COUNT PLOT
    # -------------------------------
    plt.figure(figsize=(12, 6))

    ax = sns.countplot(
        x='genre_label',
        data=features_df,
        hue='genre_label',
        palette='viridis',
        legend=False
    )

    ax.set_title('Distribution of Music Genres in the Dataset', fontsize=16)
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Number of Segments', fontsize=12)

    ax.set_xticks(range(len(genre_names)))
    ax.set_xticklabels(genre_names, rotation=30)

    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 2. BOX PLOT (Spectral Centroid)
    # -------------------------------
    print("\n--- Generating Box Plot for Spectral Centroid ---")

    plt.figure(figsize=(14, 7))

    box_ax = sns.boxplot(
        x='genre_label',
        y='25',
        data=features_df,
        hue='genre_label',
        palette='cubehelix',
        legend=False
    )

    box_ax.set_title('Spectral Centroid Distribution Across Genres', fontsize=18)
    box_ax.set_xlabel('Genre', fontsize=14)
    box_ax.set_ylabel('Spectral Centroid', fontsize=14)

    box_ax.set_xticks(range(len(genre_names)))
    box_ax.set_xticklabels(genre_names, rotation=30, ha="right")

    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 3. VIOLIN PLOT (MFCC 1)
    # -------------------------------
    print("\n--- Generating Violin Plot for First MFCC (Column 0) ---")

    plt.figure(figsize=(14, 7))

    violin_ax = sns.violinplot(
        x='genre_label',
        y='0',
        data=features_df,
        hue='genre_label',
        palette='Spectral',
        legend=False
    )

    violin_ax.set_title('First MFCC (Timbre/Energy) Distribution Across Genres', fontsize=18)
    violin_ax.set_xlabel('Genre', fontsize=14)
    violin_ax.set_ylabel('MFCC 1 Value', fontsize=14)

    violin_ax.set_xticks(range(len(genre_names)))
    violin_ax.set_xticklabels(genre_names, rotation=30, ha="right")

    plt.tight_layout()
    plt.show()

    # =====================================================

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure you have run the 'feature_extractor.py' script first to generate the dataset.")

except Exception as e:
    print(f"An error occurred while loading the DataFrame: {e}")