# Search and evidence scope

Date: 2026-10-25. Nature academic search workflow: multi-source discovery and primary-source validation. Dedicated academic-search MCP endpoints were not exposed; web search, PubMed-indexed primary records, publisher pages, NCBI protein FASTA, and the Europe PMC full-text and supplementary-file APIs were used. Discovery was targeted, not a claim of exhaustive systematic coverage.

Queries included mycotoxin degradation enzyme 2026 2025 kinetic characterization Km kcat sequence; zearalenone hydrolase 2026 2025 Km kcat enzyme; aflatoxin degrading enzyme 2026 2025 kinetic kcat Km; 2025 2026 ochratoxin hydrolase Km kcat; 2025 patulin enzyme Km kcat; 2026 zearalenone hydrolase kinetic novel enzyme; and follow-up exact enzyme names, DOIs, accessions, kinetic parameters and Table identifiers.

Searches broadened to 2023–2024 papers absent from the original kinetic training table. The year in the ledger is the bibliographic year except MbAmh1, for which online 2024 versus issue 2025 is explicitly stated. An enzyme published before 2026 can be new to the training data without being a newly published discovery. DOI and identical sequence checks were used to avoid presenting a known training example as independent.

Primary evidence anchors:

- ZenA: ACS Catalysis DOI 10.1021/acscatal.4c00271; PMC10913051. Table 1 is a raster table; the original image was visually read. Every one of the nine characterized, soluble enzymes is retained; insoluble homologues lack measured kinetic values.
- ZHRnZ: DOI 10.3390/ijms25179665; PMC11395237, Table 3; GAP90066.2. E122 substitutions verified in the reference sequence before reconstruction.
- ZPH1101: DOI 10.3390/toxins17010021; PMC11768653. The primary full text explicitly states 100% identity with AFQ59371.
- ZHDAY3: DOI 10.1128/aem.01818-23; PMC10952378; primary publisher full-text indexed content and Table 3 at https://journals.asm.org/doi/full/10.1128/aem.01818-23. The primary text explicitly identifies XP_013255140.1 and 264 residues. WT and N153H kinetic values are for alpha-zearalanol. Structural entry 8JLV contains S102A and was not substituted for kinetic WT. Europe PMC full-text retrieval returned HTTP 500 and the NCBI BioC endpoint returned no result; publicly indexed publisher/PMC primary content supplied the evidence.
- AsPLa-D: DOI 10.1002/pro.70729; PMC13413597, Figure 5 and author SI. The designed amino acid sequence is copied from SI; this does not resolve the MBP-fusion assay effect.
- MgSDR1: DOI 10.1002/advs.202501146; PMC12225012, author SI Table S5. The accession label in Figure 3A has NP rather than XP. XP_001483215.1 is length- and motif-consistent; this correction remains a disclosed reference-sequence mapping rather than a newly verified experimental-strain sequence.

Some publisher and figshare downloads returned 403; some MDPI requests returned 429. Europe PMC full-text/SI APIs supplied accessible open-access primary copies where available. Cases without a verified sequence/target pair were marked pending or not predicted. For example, KFL25551 is the DepA sequence used to design DDE primers, not a confirmed sequence for the experimentally measured DDE from Devosia JA3.

Do not silently replace inaccessible tested sequences with homologues. Do not equate specific activity, percent degradation, stability, colorimetric TMB kinetics, or nanoparticle catalytic rates with a protein–mycotoxin kcat/Km measurement. Do not count multiple variants from a study as independent publications.
