# Input-space corruption definitions

Let L=512 and n=floor(rho L). All sequence operators preserve length. Replacement bytes are uniform integers0..255. All positions, including header/masked/padding positions, are eligible unless preserve_prefix is explicitly set. No operator constructs protocol-valid adversarial traffic.

- INP: overwrite the centered interval[start,start+n),start=floor((L-n)/2).
- IBP: independently select each position with probability rho and replace selected positions; realized selected count is random.
- RTO: overwrite the final n positions with random bytes.
- CTO: overwrite the final n positions with constant255.
- Gaussian: add rho*standard_normal to training-standardized statistics. Its rho is a standard deviation, not a byte replacement fraction.
- Joint: the public structural preset applies INP and Gaussian to the same Query. Prefix protection is explicit and defaults0. Historical54-byte-protected or IBP-joint protocols are different named configurations and must not be conflated.

For matched IBP evaluation the RNG draws the mask before replacement bytes, keyed by dataset/raw sampleID. Noisy reconstruction training draws replacement values before its mask and alternates odd-step IBP30/even-step INP30, Query-only and both labels. Rec0/rec0.1 share the complete training structure, retaining the training decoder in both; only its loss coefficient changes. Clean-trained Concatenation is a separate control.

Historical APR/IDP names have multiple meanings across retained scripts. This release defines RTO/CTO operationally; renaming does not certify every historical CSV. The canonical four-operator source uses fixed-length overwrite, not physical appending or dummy-packet insertion.

adaptive_pgd.py separately provides the retained exact-discrete-forward, embedding-interpolation BPDA attack. It uses unsquared prototype logits by design; do not silently apply it to a squared-distance classifier and claim identical attack semantics. Random restarts are optimization starts, not independent training seeds; byte modification eligibility is not physical packet legality.
