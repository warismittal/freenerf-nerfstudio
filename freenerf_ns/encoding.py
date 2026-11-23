"""NeRF positional encoding with the FreeNeRF frequency mask. Requires nerfstudio + torch."""
from __future__ import annotations

import torch
from nerfstudio.field_components.encodings import NeRFEncoding

from .mask import band_groups, check_layout, group_weights


class MaskedNeRFEncoding(NeRFEncoding):
    """nerfstudio's NeRFEncoding times a training-progress mask alpha(t) (paper Eq. 6).

    The mask is stored as persistent buffers (step, end), so a checkpoint rendered later restores the
    exact mask it was trained with. It applies to positions AND directions (one instance of each is shared by
    the coarse and fine fields, as in nerfstudio's vanilla NeRF).
    """

    def __init__(self, in_dim: int = 3, num_frequencies: int = 10, min_freq_exp: float = 0.0, max_freq_exp: float = 8.0,
                 include_input: bool = True, mask_mode: str = "none", reverse: bool = False) -> None:
        super().__init__(in_dim=in_dim, num_frequencies=num_frequencies, min_freq_exp=min_freq_exp,
                         max_freq_exp=max_freq_exp, include_input=include_input)
        self.mask_mode = mask_mode
        groups, n_groups = band_groups(in_dim, num_frequencies, include_input, reverse)
        self.register_buffer("groups", torch.as_tensor(groups), persistent=False)
        self.n_groups = n_groups
        self.register_buffer("step_buf", torch.zeros((), dtype=torch.long))
        self.register_buffer("end_buf", torch.ones((), dtype=torch.long))
        self._py_step, self._py_end = 0, 1
        self._cache = (None, None)

        def _plain(x):  # bypass the mask: the raw nerfstudio encoding, for the layout self-check
            with torch.no_grad():
                return NeRFEncoding.forward(self, torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()

        check_layout(_plain, in_dim, num_frequencies, include_input)  # fails loudly if nerfstudio changed its layout

    def set_progress(self, step: int, end_step: int | None = None) -> None:
        self._py_step = int(step)
        if end_step is not None:
            self._py_end = int(end_step)
        self.step_buf.fill_(self._py_step)
        self.end_buf.fill_(self._py_end)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._py_step, self._py_end = int(self.step_buf.item()), int(self.end_buf.item())
        self._cache = (None, None)

    def forward(self, in_tensor, *args, **kwargs):
        enc = super().forward(in_tensor, *args, **kwargs)
        if self.mask_mode == "none":
            return enc
        key = (self._py_step, self._py_end, str(enc.device))
        if self._cache[0] != key:
            w = torch.as_tensor(group_weights(self._py_step, self._py_end, self.n_groups, self.mask_mode)).to(enc.device)
            self._cache = (key, w[self.groups])
        return enc * self._cache[1].to(enc.dtype)
