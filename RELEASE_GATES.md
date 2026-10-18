# Checks after the public repository push, before archival or submission

1. The author selected MIT for project-authored code. Review `SOURCE_AND_LICENSE_REVIEW.md`; PIQA and C4 permissions should not be inferred from an ambiguous dataset card. The package excludes upstream weights and benchmark text, but review of derived sample metrics and attribution remains necessary.
2. Review the generated CSVs and protocol text for disclosure of unpublished dataset sample IDs, internal node names, and author affiliations. The automated scan checks common absolute paths and credential patterns; it cannot replace review.
3. Public repository: https://github.com/extradimen/layer-pruning-operating-regions . Freeze an archival commit, assign a persistent identifier, and update the manuscript's Data and Code Availability statement to the actual URL/DOI and exact accessible scope.
4. Decide whether the 23 MB private evidence bundle should be provided to editors/reviewers through controlled access. It contains operational logs and paths and is not part of this public package.
5. The public clone passed `python3 verify_release.py`. Both recomputation scripts passed in the governed source environment, and figure regeneration passed in a clean copy. Before persistent archiving, repeat the full check at the archival commit and record the archive checksum in the submission record.

The package is public. These remaining checks concern archival and manuscript submission.
