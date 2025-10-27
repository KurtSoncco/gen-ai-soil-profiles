import numpy as np
import torch


class TTSDataset(torch.utils.data.Dataset):
    """
    Dataset for standardized TTS profiles.
    Optionally returns a corrupted (noisy) input with the clean target for DAE training.
    """

    def __init__(
        self,
        data: np.ndarray,
        corruption_noise_std: float | None = None,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.data = torch.from_numpy(data).float()
        self.corruption_noise_std = corruption_noise_std
        self.generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        clean = self.data[idx]
        if self.corruption_noise_std is None or self.corruption_noise_std <= 0:
            return clean, clean

        noise = torch.randn_like(clean) * self.corruption_noise_std
        noisy = clean + noise
        return noisy, clean
