import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CONTROL = "verinuddle_H3LatentControl";
const SAVE = "verinuddle_SaveLatent";
const LOAD = "verinuddle_LoadLatent";
const MAX_INDEX = 9999;
const DEFAULT_INDEX = 1;

const CSS = `
.verinuddle-h3-control {
    display: flex;
    flex-direction: column;
    gap: 6px;
    width: 100%;
    box-sizing: border-box;
    padding: 2px 0 4px;
}

.verinuddle-h3-control-row {
    display: flex;
    gap: 6px;
}

.verinuddle-h3-control-row button {
    flex: 1;
    cursor: pointer;
    border: 1px solid #555;
    background: #2a2a2a;
    color: #ddd;
    border-radius: 4px;
    padding: 6px 4px;
    font: 12px sans-serif;
}

.verinuddle-h3-control-row button:hover {
    background: #3a3a3a;
}

.verinuddle-h3-control-meta {
    opacity: 0.75;
    font-size: 11px;
    line-height: 1.3;
    min-height: 14px;
    word-break: break-word;
}
`;

let cssInjected = false;

function injectCss() {
    if (cssInjected) return;
    cssInjected = true;

    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
}

function swallow(element) {
    for (const type of [
        "pointerdown",
        "mousedown",
        "pointerup",
        "mouseup",
        "click",
        "dblclick",
        "contextmenu",
    ]) {
        element.addEventListener(type, (event) => event.stopPropagation());
    }
}

function graphNodes(graph) {
    return graph?._nodes || graph?.nodes || [];
}

function groupList(graph) {
    return graph?._groups || graph?.groups || [];
}

function groupMembers(group) {
    if (typeof group.recomputeInsideNodes === "function") {
        try {
            group.recomputeInsideNodes();
        } catch (_) {
            // Older LiteGraph versions may not support this.
        }
    }

    if (Array.isArray(group._nodes) && group._nodes.length) {
        return group._nodes;
    }

    if (Array.isArray(group.nodes) && group.nodes.length) {
        return group.nodes;
    }

    const children = group._children || group.children;
    if (children) {
        return Array.from(children);
    }

    return [];
}

function inGroup(group, node) {
    const members = groupMembers(group);

    if (
        members.includes(node) ||
        members.some((member) => member.id === node.id)
    ) {
        return true;
    }

    const bounds = group._bounding || group.bounding;
    if (!bounds || bounds.length < 4 || !node.pos) {
        return false;
    }

    const x = node.pos[0] + (node.size?.[0] || 0) / 2;
    const y = node.pos[1] + (node.size?.[1] || 0) / 2;

    return (
        x >= bounds[0] &&
        x <= bounds[0] + bounds[2] &&
        y >= bounds[1] &&
        y <= bounds[1] + bounds[3]
    );
}

function findPair(control) {
    const graph = control.graph || app.graph;
    const nodes = graphNodes(graph);

    const loads = nodes.filter((node) => node.comfyClass === LOAD);
    const saves = nodes.filter((node) => node.comfyClass === SAVE);

    // Prefer a Save/Load pair in the same canvas group as the control node.
    for (const group of groupList(graph)) {
        if (!inGroup(group, control)) continue;

        const members = groupMembers(group);
        const pool = members.length
            ? members
            : nodes.filter((node) => inGroup(group, node));

        const load = pool.find((node) => node.comfyClass === LOAD);
        const save = pool.find((node) => node.comfyClass === SAVE);

        if (load && save) {
            return { load, save };
        }
    }

    // If there is exactly one Save and one Load in the graph, use them.
    if (loads.length === 1 && saves.length === 1) {
        return {
            load: loads[0],
            save: saves[0],
        };
    }

    return null;
}

function widgetValue(node, name) {
    return node?.widgets?.find((widget) => widget.name === name)?.value;
}

function indexWidget(node) {
    return node?.widgets?.find((widget) => widget.name === "index");
}

function pathWidget(node) {
    return node?.widgets?.find((widget) => widget.name === "path");
}

function readIndex(node) {
    const widget = indexWidget(node);
    const value = Number.parseInt(widget?.value, 10);

    if (!Number.isFinite(value)) {
        return DEFAULT_INDEX;
    }

    return Math.max(0, Math.min(MAX_INDEX, value));
}

function readPath(node) {
    const value = widgetValue(node, "path");

    if (value == null) {
        return "";
    }

    return String(value).trim();
}

function writeIndex(node, value) {
    const widget = indexWidget(node);

    if (!widget) {
        return false;
    }

    const next = Math.max(
        0,
        Math.min(MAX_INDEX, Number.parseInt(value, 10) || 0)
    );

    widget.value = next;

    if (typeof widget.callback === "function") {
        widget.callback(next);
    }

    return true;
}

function setStatus(control, message) {
    if (control._verinuddleH3?.meta) {
        control._verinuddleH3.meta.textContent = message;
    }
}

function refreshStatus(control) {
    const pair = findPair(control);

    if (!pair) {
        setStatus(
            control,
            "Save, Load, and Control must share one canvas group."
        );
        return;
    }

    const loadIndex = readIndex(pair.load);
    const saveIndex = readIndex(pair.save);

    setStatus(control, `Load ${loadIndex} / Save ${saveIndex}`);
}

function updateIndices(control, delta) {
    const pair = findPair(control);

    if (!pair) {
        refreshStatus(control);
        return;
    }

    const loadIndex = readIndex(pair.load);
    const saveIndex = readIndex(pair.save);

    writeIndex(pair.load, loadIndex + delta);
    writeIndex(pair.save, saveIndex + delta);

    app.graph?.setDirtyCanvas?.(true, true);
    refreshStatus(control);
}

function resetIndices(control) {
    const pair = findPair(control);

    if (!pair) {
        refreshStatus(control);
        return;
    }

    writeIndex(pair.load, DEFAULT_INDEX - 1);
    writeIndex(pair.save, DEFAULT_INDEX);

    app.graph?.setDirtyCanvas?.(true, true);
    refreshStatus(control);
}

async function clearLatents(control) {
    const pair = findPair(control);

    if (!pair) {
        refreshStatus(control);
        return;
    }

    const savePath = readPath(pair.save);
    const loadPath = readPath(pair.load);

    if (!savePath || !loadPath) {
        setStatus(control, "Save and Load paths must both be set.");
        return;
    }

    if (savePath !== loadPath) {
        setStatus(control, "Save and Load paths differ. Nothing was deleted.");
        return;
    }

    try {
        const response = await api.fetchApi("/verinuddle/clear_latents", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                path: savePath,
            }),
        });

        if (!response.ok) {
            setStatus(
                control,
                `Clear failed (HTTP ${response.status}).`
            );
            return;
        }

        const result = await response.json();
        const removed = Number.parseInt(result.removed, 10) || 0;

        setStatus(
            control,
            removed
                ? `Removed ${removed} latent${removed === 1 ? "" : "s"}.`
                : "No matching latents found."
        );
    } catch (error) {
        console.error("Verinuddle H3 Latent Control:", error);
        setStatus(control, "Clear failed.");
    }
}

app.registerExtension({
    name: "verinuddle.h3_latent_control",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== CONTROL) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);

            injectCss();

            const root = document.createElement("div");
            root.className = "verinuddle-h3-control";

            const row1 = document.createElement("div");
            row1.className = "verinuddle-h3-control-row";

            const row2 = document.createElement("div");
            row2.className = "verinuddle-h3-control-row";

            const increaseButton = document.createElement("button");
            increaseButton.textContent = "Increase";
            increaseButton.title = "Increase both Save and Load indices by 1.";

            const decreaseButton = document.createElement("button");
            decreaseButton.textContent = "Decrease";
            decreaseButton.title =
                "Decrease both Save and Load indices by 1. Minimum is 0.";

            const resetButton = document.createElement("button");
            resetButton.textContent = "Reset";
            resetButton.title =
                "Reset both Save and Load indices to their default value of 1.";

            const clearButton = document.createElement("button");
            clearButton.textContent = "Clear latents";
            clearButton.title =
                "Delete only <path>_<index>.safetensors files for the current Save/Load path.";

            const meta = document.createElement("div");
            meta.className = "verinuddle-h3-control-meta";

            row1.append(increaseButton, decreaseButton);
            row2.append(resetButton, clearButton);
            root.append(row1, row2, meta);

            swallow(root);

            this.addDOMWidget(
                "h3_latent_control",
                "CONTROL",
                root,
                { serialize: false }
            );

            this._verinuddleH3 = {
                meta,
            };

            increaseButton.onclick = (event) => {
                event.stopPropagation();
                updateIndices(this, 1);
            };

            decreaseButton.onclick = (event) => {
                event.stopPropagation();
                updateIndices(this, -1);
            };

            resetButton.onclick = (event) => {
                event.stopPropagation();
                resetIndices(this);
            };

            clearButton.onclick = async (event) => {
                event.stopPropagation();
                await clearLatents(this);
            };

            refreshStatus(this);

            this.setSize?.([270, 120]);

            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;

        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            refreshStatus(this);
            return result;
        };
    },
});
