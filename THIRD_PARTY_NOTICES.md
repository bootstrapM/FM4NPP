# Third-party provenance

This repository is an independent inference harness for FM4NPP. It does not
contain model weights or dataset events.

The checkpoint-compatible architecture was implemented from:

- FM4NPP paper: <https://arxiv.org/abs/2508.14087>
- FM4NPP publication code, inspected at commit
  `2cd116b493aba86faed0164bbb6c140b35bd512d`:
  <https://github.com/FM4NPP/PP_collision>
- The official Mamba implementation and Mamba2 selective state-space
  recurrence: <https://github.com/state-spaces/mamba>

The Hugging Face model card labels the checkpoints `lgpl`. The upstream GitHub
repository did not include a license file when the implementation above was
prepared. Confirm the terms that apply to your use before redistributing an
adapted model or upstream source code.

The optional preprocessing statistics in
`src/fm4npp_inference/data/tpcpp10m_test_bins.json` were computed from the
public TPCpp-10M labeled test split (Zenodo record `16970029`), which is
published under CC BY 4.0. They are an approximation to the unavailable
training-time bin-edge file and are labeled as such in outputs.

