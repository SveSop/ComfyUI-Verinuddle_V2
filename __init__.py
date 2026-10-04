WEB_DIRECTORY = "./web"

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .latent_io import (
    SaveLatentPath,
    LoadLatentPath,
    BackupLatentPath,
    SaveLatent,
    LoadLatent,
    H3LatentControl,
    register_latent_control_routes,
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
            H3LatentControl,
        ]


async def comfy_entrypoint() -> VerinuddleExtension:
    register_latent_control_routes()
    return VerinuddleExtension()