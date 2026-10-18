# Checks before a public repository push

1. The author selected MIT for project-authored code. Review `SOURCE_AND_LICENSE_REVIEW.md`; PIQA and C4 permissions should not be inferred from an ambiguous dataset card. The package excludes upstream weights and benchmark text, but review of derived sample metrics and attribution remains necessary.
2. Review the generated CSVs and protocol text for disclosure of unpublished dataset sample IDs, internal node names, and author affiliations. The automated scan checks common absolute paths and credential patterns; it cannot replace review.
3. Freeze a public commit, archive it with a persistent identifier, and update the manuscript's Data and Code Availability statement to the actual URL/DOI and exact accessible scope. Do not claim public availability before the deposit exists.
4. Decide whether the 23 MB private evidence bundle should be provided to editors/reviewers through controlled access. It contains operational logs and paths and is not part of this public package.
5. Run `python3 verify_release.py` in a clean copy, then execute both recomputation scripts and the figure script in the declared analysis environment before deposit. Record the resulting commit and archive checksum in the submission record.

The current package is therefore ready for technical review, not an authorization to publish it.
