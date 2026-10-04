from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .latent_io import SaveLatentPath, LoadLatentPath, BackupLatentPath, SaveLatent, LoadLatent
from .conditioning_io import (
    SaveConditioningPath,
    LoadConditioningPath,
    BackupConditioningPath,
    SaveConditioning,
    LoadConditioning,
)


class VerinuddleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveLatentPath,
            LoadLatentPath,
            BackupLatentPath,
            SaveLatent,
            LoadLatent,
            SaveConditioningPath,
            LoadConditioningPath,
            BackupConditioningPath,
            SaveConditioning,
            LoadConditioning,
        ]


async def comfy_entrypoint() -> VerinuddleExtension:
    return VerinuddleExtension()
