from NMAP.connectome_priors.swimmer_priors import generate_ncap_segment_priors, refresh_inventory_files


def test_refresh_inventory_files_smoke():
    metadata = refresh_inventory_files()
    assert isinstance(metadata, dict)
    assert "warnings" in metadata


def test_generate_ncap_segment_priors_sparse_scalars():
    priors = generate_ncap_segment_priors(num_segments=8)

    required = (
        "dist_ipsi_db",
        "dist_ipsi_vb",
        "dist_contra_db",
        "dist_contra_vb",
        "dist_next_db",
        "dist_next_vb",
        "syn_ipsi_db",
        "syn_ipsi_vb",
        "syn_contra_db",
        "syn_contra_vb",
        "syn_next_db",
        "syn_next_vb",
        "count_ipsi_db",
        "count_ipsi_vb",
        "count_contra_db",
        "count_contra_vb",
        "count_next_db",
        "count_next_vb",
    )
    for key in required:
        assert key in priors

    for key in ("dist_ipsi_db", "dist_ipsi_vb", "dist_contra_db", "dist_contra_vb", "dist_next_db", "dist_next_vb"):
        assert 0.0 <= float(priors[key]) <= 1.0
    for key in ("count_ipsi_db", "count_ipsi_vb", "count_contra_db", "count_contra_vb", "count_next_db", "count_next_vb"):
        assert int(priors[key]) >= 0
