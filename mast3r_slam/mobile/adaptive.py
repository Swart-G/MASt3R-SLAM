from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class TransportProfile:
    jpeg_quality: int = 80
    fps: int = 10


class AdaptiveBitrate:
    """Conservative controller optimized for freshness rather than throughput."""

    def __init__(self, profile: TransportProfile = TransportProfile()) -> None:
        self.profile = profile

    def update(self, *, latency_ms: int, dropped: bool) -> TransportProfile:
        quality, fps = self.profile.jpeg_quality, self.profile.fps
        overloaded = dropped or latency_ms >= 900
        free = not dropped and latency_ms <= 350
        if overloaded:
            if fps > 3:
                fps = max(3, fps - 1)
            elif quality > 50:
                quality = max(50, quality - 5)
        elif free:
            if quality < 90:
                quality = min(90, quality + 2)
            elif fps < 15:
                fps += 1
        self.profile = TransportProfile(quality, fps)
        return self.profile
