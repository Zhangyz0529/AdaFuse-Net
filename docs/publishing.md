# Publishing this release

This directory and its archive are local release artifacts. They are not a public repository, DOI or anonymous review link. To make a public-availability statement, upload the audited source to the chosen hosting service, create a fixed version tag, and insert the real URL in the manuscript and citation metadata. Do not claim a URL exists before it is accessible.

Before upload, run scripts/audit_release.py and the tests. Package only source/configuration/docs and empty data/output roots. Exclude Python caches, datasets, checkpoints and runtime outputs. SOURCE_MANIFEST.json identifies the code/configuration snapshot. Record the released archive hash outside the repository. Dataset terms are separate from the project code license.
