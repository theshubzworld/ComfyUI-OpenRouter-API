import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    CHOOSE_MODEL,
    NO_MODEL,
    CATALOG_ERROR,
    MEDIA_INPUTS,
    MEDIA_SPECS,
    desiredMediaInputNames,
    ensureApiKeyWidget,
    migrateLegacyWidgetValues,
    requiredModalities,
    compatibleModels,
    nextModelValue,
} from "./model_filter.mjs";

const NODE_ID = "OpenRouterSimple";

let catalogPromise = null;

async function loadCatalog() {
    if (!catalogPromise) {
        catalogPromise = api.fetchApi("/openrouter_simple/models", { cache: "no-store" })
            .then(async (response) => {
                const payload = await response.json();
                if (!response.ok || !Array.isArray(payload.models)) {
                    throw new Error(payload.warning || `model metadata returned HTTP ${response.status}`);
                }
                return payload;
            })
            .catch((error) => {
                catalogPromise = null;
                throw error;
            });
    }
    return catalogPromise;
}

async function refreshModels(node) {
    const widget = node.widgets?.find((candidate) => candidate.name === "model");
    if (!widget) return;

    try {
        const catalog = await loadCatalog();
        const compatible = compatibleModels(catalog.models, requiredModalities(node.inputs));
        const values = compatible.length ? [CHOOSE_MODEL, ...compatible] : [NO_MODEL];
        widget.options = widget.options || {};
        widget.options.values = values;
        widget.value = compatible.length ? nextModelValue(widget.value, compatible) : NO_MODEL;
        widget.label = catalog.stale
            ? `model (${compatible.length} compatible, cached metadata)`
            : `model (${compatible.length} compatible)`;
        node.openRouterCatalogWarning = catalog.warning || null;
        node.openRouterCatalogRetried = false;
    } catch (error) {
        widget.options = widget.options || {};
        widget.options.values = [CATALOG_ERROR];
        widget.value = CATALOG_ERROR;
        widget.label = "model (catalog unavailable)";
        node.openRouterCatalogWarning = String(error?.message || error);
        if (!node.openRouterCatalogRetried) {
            node.openRouterCatalogRetried = true;
            setTimeout(() => void refreshModels(node), 2000);
        }
    }
    node.graph?.setDirtyCanvas(true, true);
}

function syncProgressiveInputs(node) {
    const desired = desiredMediaInputNames(node.inputs || []);
    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index -= 1) {
        const input = node.inputs[index];
        if (MEDIA_INPUTS.has(input?.name) && !desired.has(input.name) && input.link == null) {
            node.removeInput(index);
        }
    }
    for (const spec of MEDIA_SPECS) {
        for (const name of spec.names) {
            if (desired.has(name) && !node.inputs?.some((input) => input.name === name)) {
                node.addInput(name, spec.type);
            }
        }
    }
    node.graph?.setDirtyCanvas(true, true);
}

function scheduleNodeUpdate(node) {
    if (node.openRouterUpdateScheduled) return;
    node.openRouterUpdateScheduled = true;
    queueMicrotask(() => {
        node.openRouterUpdateScheduled = false;
        ensureApiKeyWidget(node);
        syncProgressiveInputs(node);
        void refreshModels(node);
    });
}

app.registerExtension({
    name: "OpenRouterSimple.ModalityModelFilter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            ensureApiKeyWidget(this);
            const result = onNodeCreated?.apply(this, arguments);
            scheduleNodeUpdate(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            ensureApiKeyWidget(this);
            migrateLegacyWidgetValues(arguments[0]);
            const result = onConfigure?.apply(this, arguments);
            scheduleNodeUpdate(this);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slotIndex, connected, linkInfo, slot) {
            const slotName = slot?.name || this.inputs?.[slotIndex]?.name;
            const result = onConnectionsChange?.apply(this, arguments);
            if (slotType === 1 && MEDIA_INPUTS.has(slotName)) {
                scheduleNodeUpdate(this);
            }
            return result;
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const result = getExtraMenuOptions?.apply(this, arguments);
            options.push({
                content: "Set / Fill OpenRouter API Key",
                callback: () => {
                    const widget = ensureApiKeyWidget(this) || this.widgets?.find((w) => w.name === "api_key");
                    const current = widget ? widget.value : "";
                    const entered = window.prompt("Enter OpenRouter API Key (sk-or-v1-...):", current);
                    if (entered !== null) {
                        if (widget) {
                            widget.value = entered.trim();
                        }
                        this.setDirtyCanvas(true, true);
                    }
                },
            });
            options.push({
                content: "Load Selected Preset into System Prompt Textbox",
                callback: async () => {
                    const presetWidget = this.widgets?.find((w) => w.name === "system_preset");
                    const promptWidget = this.widgets?.find((w) => w.name === "system_prompt");
                    if (!presetWidget || !promptWidget) return;
                    const selectedPreset = presetWidget.value || "Ref2VA";
                    try {
                        const res = await api.fetchApi("/openrouter_simple/presets");
                        const data = await res.json();
                        const text = data?.prompts?.[selectedPreset] || "";
                        promptWidget.value = text;
                        this.setDirtyCanvas(true, true);
                    } catch (err) {
                        console.error("Failed to load preset prompt:", err);
                    }
                },
            });
            return result;
        };
    },
});
