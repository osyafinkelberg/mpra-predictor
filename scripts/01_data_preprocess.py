from pathlib import Path
import numpy as np
import polars as pl


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

STRAINS_DSDNA = [
    'Herpesvirus:Human_cytomegalovirus', 'Herpesvirus:Epstein_Barr_Virus',
    'Herpesvirus:6B', 'Herpesvirus:Human_Herpes_7', 'Herpesvirus:Kaposi_Sarcoma_[HHV-8]',
    'Herpesvirus:Herpes_Simplex_1_KOS', 'Herpesvirus:Varicella_Zoster_Virus_Ellen_Strain',
    'Herpesvirus:Herpes_Simplex_2_Strain_G', 'Adenovirus:Type_4_Strain_RI-67',
    'Adenovirus:Type_5_Strain_Adenoid_75', 'Adenovirus:Type_1_Strain_Adenoid_71',
    'Adenovirus:Type_3_Strain_GB', 'Adenovirus:Type_37_Strain_GW_[76-19026]',
    'Adenovirus:Type_7_Strain_Gomen', 'Adenovirus:Type_14_Strain_de_Wit',
    'Adenovirus:Type_11_Strain_Slobitski', 'Papilloma_Virus:Type_2', 'Papilloma_Virus:Type_11',
    'Papilloma_Virus:Type_6b', 'Papilloma_Virus:Type_5', 'Papilloma_Virus:Type_1',
    'Papilloma_Virus:Type_52', 'Papilloma_Virus:Type_18', 'Papilloma_Virus:Type_16',
    'Polyomavirus:Merkel_Cell_Polyoma_Strain_MKL-1', 'Polyomavirus:BK_Strain_MM',
    'Polyomavirus:JC_Strain_MAD-4'
]

STRAINS_RETRO = [
    'HIV-1:CH058', 'HIV-2:GH-1', 'HIV-2:ROD', 'HIV-1:REJO',
    'Human_T-Lymphoma_Virus:Type_3', 'Human_T-Lymphoma_Virus:Type_2',
    'Human_T-Lymphoma_Virus:Type_1',
]


def generate_bootstrap_distribution(
    count_values: np.ndarray, n_boot_reps: int, random_seed: int = 0
) -> np.ndarray:
    n_tiles, n_reps = count_values.shape
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(0, n_reps, size=(n_tiles, n_boot_reps, n_reps))
    boot_samples = np.take_along_axis(np.expand_dims(count_values, 1), sample_indices, axis=2)
    return boot_samples  # (n-tiles, n-bootstrap_reps, n-replicates)


def generate_lfc_distribution(  # 'lfc': Log2-Fold Change, 'lfcp': Log2-Fold Change Plus 1
    dna_values: np.ndarray, rna_values: np.ndarray,
    n_boot_reps: int = 1000, random_seed: int = 0
) -> np.ndarray:
    assert dna_values.shape[0] == rna_values.shape[0], "Number of tiles (rows) must match between DNA and RNA arrays."
    rng = np.random.default_rng(random_seed)
    dna_seed = rng.integers(low=1, high=100, size=1)
    rna_seed = rng.integers(low=1, high=100, size=1)
    dna_boot_samples = generate_bootstrap_distribution(dna_values, n_boot_reps, dna_seed)
    dna_mean_samples = np.nanmean(dna_boot_samples, axis=2)  # (n-tiles, n-bootstrap_reps)
    rna_boot_samples = generate_bootstrap_distribution(rna_values, n_boot_reps, rna_seed)
    rna_mean_samples = np.nanmean(rna_boot_samples, axis=2)  # (n-tiles, n-bootstrap_reps)
    lfc_samples = np.log2(rna_mean_samples / dna_mean_samples)
    return lfc_samples  # (n-tiles, n-bootstrap_reps)


def generate_lfc_variance_distribution(
    dna_values: np.ndarray, rna_values: np.ndarray,
    n_boot_reps: int = 1000, random_seed: int = 0
) -> np.ndarray:
    n_tiles = dna_values.shape[0]
    assert n_tiles == rna_values.shape[0], "Number of tiles (rows) must match between DNA and RNA arrays."
    rng = np.random.default_rng(random_seed)
    dna_seed = rng.integers(low=1, high=100, size=1)
    rna_seed = rng.integers(low=1, high=100, size=1)
    dna_boot_samples = generate_bootstrap_distribution(dna_values, n_boot_reps, dna_seed)
    rna_boot_samples = generate_bootstrap_distribution(rna_values, n_boot_reps, rna_seed)
    lfc_matrix = np.log2(rna_boot_samples[:, :, :, None]) - np.log2(dna_boot_samples[:, :, None, :])
    lfc_values = lfc_matrix.reshape(n_tiles, n_boot_reps, -1)  # (n-tiles, n-bootstrap_reps, n-dna_reps * n-rna_reps)
    lfc_variance = np.nanvar(lfc_values, axis=-1, ddof=1)  # (n-tiles, n-bootstrap_reps)
    return lfc_variance  # (n-tiles, n-bootstrap_reps)


def bootstrap_mean_estimation(raw_df: pl.DataFrame) -> pl.DataFrame:
    dna_columns = [col for col in raw_df.columns if col.startswith('Plasmid_r')]
    id_columns = ['ID', 'strand', 'project', 'sequence']
    dna_values = raw_df.select(dna_columns).to_numpy()
    formatted = raw_df.select(id_columns)
    formatted = formatted.with_columns(
        pl.col('ID').str.split(':').list.slice(0, 2).list.join(':').alias('strain')
    )
    mask = ~formatted['strain'].is_in(STRAINS_DSDNA + STRAINS_RETRO)
    formatted = formatted.with_columns(
        pl.when(mask)
        .then(pl.col('project'))
        .otherwise(pl.col('strain'))
        .alias('strain')
    )

    dna_boot_dist = generate_bootstrap_distribution(dna_values, n_boot_reps=1000)
    dna_mean_dist = np.nanmean(dna_boot_dist, axis=2)  # (n-tiles, n-bootstrap_reps)
    formatted = formatted.with_columns(
        pl.Series('dna_mean', dna_mean_dist.mean(axis=1))
    )
    experiments = np.unique([
        '_'.join(col.split('_')[:-1]) for col in raw_df.columns
        if col not in id_columns + dna_columns
    ])

    for experiment in experiments:
        n_replicates = sum([f'{experiment}_r' in col for col in raw_df.columns])
        if n_replicates == 0:
            continue

        rna_cols = [f'{experiment}_r{i}' for i in range(1, n_replicates + 1)]
        rna_values = raw_df.select(rna_cols).to_numpy()

        # point LFC
        lfc_point = np.log2(rna_values.mean(1) / dna_values.mean(1))

        # bootstrap LFC
        mean_dist = generate_lfc_distribution(dna_values, rna_values, n_boot_reps=1000)
        lfc_mean = np.nanmean(mean_dist, axis=1)
        lfc_mean_var = np.nanvar(mean_dist, axis=1, ddof=1)

        var_dist = generate_lfc_variance_distribution(dna_values, rna_values, n_boot_reps=1000)
        lfc_var = np.nanmean(var_dist, axis=1)
        lfc_var_var = np.nanvar(var_dist, axis=1, ddof=1)

        formatted = formatted.with_columns([
            pl.Series(f'lfc_point_{experiment}', lfc_point),
            pl.Series(f'lfc_mean_{experiment}', lfc_mean),
            pl.Series(f'lfc_mean_variance_{experiment}', lfc_mean_var),
            pl.Series(f'lfc_var_{experiment}', lfc_var),
            pl.Series(f'lfc_var_variance_{experiment}', lfc_var_var)
        ])

    return formatted


def preprocess_ol49() -> None:
    ol49_raw = pl.read_csv(DATA_DIR / "mpra_suite_processed_OL49.csv")
    for additional_experiment, experiment_name in [
        ['mpra_suite_processed_OL49_Berkay_K562_TNF.csv', 'Berkay_9/25'],
    ]:
        df = pl.read_csv(DATA_DIR / additional_experiment)
        rename_dct = {col: experiment_name + '_' + col for col in df.columns if col != 'ID'}
        df = df.rename(rename_dct)
        assert np.intersect1d(ol49_raw.columns, df.columns).tolist() == ['ID']
        ol49_raw = ol49_raw.join(df, on='ID', how='left')
    ol49_formatted = bootstrap_mean_estimation(ol49_raw)

    # train / validation / test split
    TRAIN_FRAC = 0.8
    VAL_FRAC = 0.1
    TEST_FRAC = 0.1
    assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-6, "Fractions must sum to 1"
    assert ol49_formatted['sequence'].n_unique() == ol49_formatted.height
    assert all(~ol49_formatted['sequence'].is_null())
    total_sequences = len(ol49_formatted)
    strain_counts = ol49_formatted.group_by("strain").len().rename({"len": "counts"})
    strains_shuffled = strain_counts.sample(fraction=1.0, shuffle=True, seed=42)
    train_cutoff = int(TRAIN_FRAC * total_sequences)
    val_cutoff = int((TRAIN_FRAC + VAL_FRAC) * total_sequences)

    strain_split = []
    cum_count = 0

    for row in strains_shuffled.iter_rows(named=True):
        strain = row['strain']
        count = row['counts']
        if cum_count + count <= train_cutoff:
            split = 'train'
        elif cum_count + count <= val_cutoff:
            split = 'val'
        else:
            split = 'test'
        strain_split.append((strain, split))
        cum_count += count

    strain_to_split = dict(strain_split)
    ol49_formatted = ol49_formatted.with_columns(
        pl.col('strain').replace(strain_to_split, default='test').alias('split')
    )

    retro_mask = ol49_formatted['strain'].is_in(STRAINS_RETRO)
    ol49_formatted = ol49_formatted.with_columns(
        pl.when(retro_mask & ol49_formatted['strain'].str.starts_with('HIV'))
        .then(pl.lit('val'))
        .when(retro_mask & ol49_formatted['strain'].str.starts_with('Human_T-Lymphoma_Virus'))
        .then(pl.lit('test'))
        .otherwise(pl.col('split'))
        .alias('split')
    )
    ol49_formatted.write_csv(DATA_DIR / "OL49.csv")


def preprocess_ol53() -> None:
    ol53_raw = pl.read_csv(DATA_DIR / "mpra_suite_processed_OL53.csv")
    ol53_raw = ol53_raw.filter(pl.col('dna_mean') >= 50)  # NOTE: exclude low plasmid count tiles
    ol53_formatted = bootstrap_mean_estimation(ol53_raw)

    # NOTE: this mean estimation variance implicitly depends on the library DNA amount
    ctrl_cols = [f'Jurkat_r{i}' for i in range(1, 6)]
    stim_cols = [f'JurkatStim_r{i}' for i in range(1, 6)]

    ctrl_values = ol53_raw.select(ctrl_cols).to_numpy()
    stim_values = ol53_raw.select(stim_cols).to_numpy()

    lfc_point = np.log2(stim_values.mean(1) / ctrl_values.mean(1))
    boot_dist = generate_lfc_distribution(ctrl_values, stim_values, n_boot_reps=1000)
    lfc_mean = np.nanmean(boot_dist, axis=1)
    lfc_mean_var = np.nanvar(boot_dist, axis=1, ddof=1)

    ol53_formatted = ol53_formatted.with_columns([
        pl.Series('lfc_point_StimEffect', lfc_point),
        pl.Series('lfc_mean_StimEffect', lfc_mean),
        pl.Series('lfc_mean_variance_StimEffect', lfc_mean_var),
        pl.col('ID').alias('ID_OL49')
    ])

    # match to OL49
    ol53_formatted = ol53_formatted.with_columns(
        pl.when(pl.col('project').str.starts_with('viral_'))
        .then(pl.col('ID_OL49').str.replace("_", ":").str.split(':').list.slice(0, 4).list.join(":"))
        .otherwise(pl.col('ID_OL49'))
        .alias('ID_OL49')
    )

    # train / validation / test split
    ol53_formatted = ol53_formatted.with_columns(pl.lit('train').alias('split'))
    ol53_formatted = ol53_formatted.with_columns(
        pl.when(pl.col('strain').str.starts_with('HIV'))
        .then(pl.lit('val'))
        .when(pl.col('strain').str.starts_with('Human_T-Lymphoma_Virus'))
        .then(pl.lit('test'))
        .otherwise(pl.col('split'))
        .alias('split')
    )

    ol53_formatted.write_csv(DATA_DIR / "OL53.csv")


if __name__ == "__main__":
    preprocess_ol49()
    preprocess_ol53()
