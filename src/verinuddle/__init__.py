from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .latent_io import ExportLatent, ImportLatent, VerinuddleSaveLatent, VerinuddleLoadLatent


class VerinuddleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ExportLatent,
            ImportLatent,
            VerinuddleSaveLatent,
            VerinuddleLoadLatent,
        ]


async def comfy_entrypoint() -> VerinuddleExtension:
    return VerinuddleExtension()
