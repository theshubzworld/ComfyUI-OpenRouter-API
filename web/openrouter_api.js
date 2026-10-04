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

const NODE_IDS = new Set(["OpenRouterAPI", "OpenRouterSimple"]);

let catalogPromise = null;

async function loadCatalog(force = false) {
    if (force) {
        catalogPromise = null;
    }
    if (!catalogPromise) {
        const url = force ? "/openrouter_api/models?refresh=true" : "/openrouter_api/models";
        catalogPromise = api.fetchApi(url, { cache: "no-store" })
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

async function refreshModels(node, forceCatalog = false) {
    const widget = node.widgets?.find((candidate) => candidate.name === "model");
    if (!widget) return;

    try {
        const catalog = await loadCatalog(forceCatalog);
        const freeOnly = Boolean(node.openRouterFreeOnly);
        const bypassModalities = Boolean(node.openRouterBypassModalities);
        const activeModalities = [...requiredModalities(node.inputs)].filter((m) => m !== "text");
        const required = bypassModalities ? new Set(["text"]) : requiredModalities(node.inputs);
        const compatible = compatibleModels(catalog.models, required, freeOnly);
        const values = compatible.length ? [CHOOSE_MODEL, ...compatible] : [NO_MODEL];
        widget.options = widget.options || {};
        widget.options.values = values;
        widget.value = compatible.length ? nextModelValue(widget.value, compatible) : NO_MODEL;
        const freeCount = catalog.free_count ?? catalog.models.filter((m) => m.is_free || String(m.id || "").includes(":free")).length;
        
        let filterDesc = "";
        if (freeOnly) {
            filterDesc = ` [FREE ONLY: ${compatible.length}]`;
        } else if (!bypassModalities && activeModalities.length > 0) {
            filterDesc = ` (${compatible.length} compatible with ${activeModalities.join("+")}, ${freeCount} free)`;
        } else {
            filterDesc = ` (${compatible.length} available, ${freeCount} free)`;
        }
        widget.label = catalog.stale
            ? `model${filterDesc} (cached)`
            : `model${filterDesc}`;
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
    name: "OpenRouterAPI.ModalityModelFilter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_IDS.has(nodeData.name)) return;

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
            if (arguments[0]?.openRouterFreeOnly !== undefined) {
                this.openRouterFreeOnly = Boolean(arguments[0].openRouterFreeOnly);
            }
            if (arguments[0]?.openRouterBypassModalities !== undefined) {
                this.openRouterBypassModalities = Boolean(arguments[0].openRouterBypassModalities);
            }
            const result = onConfigure?.apply(this, arguments);
            scheduleNodeUpdate(this);
            return result;
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            onSerialize?.apply(this, arguments);
            if (this.openRouterFreeOnly) {
                o.openRouterFreeOnly = true;
            }
            if (this.openRouterBypassModalities) {
                o.openRouterBypassModalities = true;
            }
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
                content: this.openRouterBypassModalities
                    ? "Input Modality Filter: BYPASSING (Click to filter by inputs)"
                    : "Input Modality Filter: ACTIVE (Click to show all 466 models)",
                callback: () => {
                    this.openRouterBypassModalities = !this.openRouterBypassModalities;
                    void refreshModels(this);
                },
            });
            options.push({
                content: this.openRouterFreeOnly
                    ? "Show All Models (Free & Paid)"
                    : "Filter: Show Free Models Only (:free)",
                callback: () => {
                    this.openRouterFreeOnly = !this.openRouterFreeOnly;
                    void refreshModels(this);
                },
            });
            options.push({
                content: "Fetch & Refresh Live Models from OpenRouter",
                callback: async () => {
                    await refreshModels(this, true);
                },
            });
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
                        const res = await api.fetchApi("/openrouter_api/presets");
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
