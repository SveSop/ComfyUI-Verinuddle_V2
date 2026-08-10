from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class VerinuddleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return []


async def comfy_entrypoint() -> VerinuddleExtension:
    return VerinuddleExtension()
