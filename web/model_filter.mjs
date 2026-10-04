export const CHOOSE_MODEL = "— choose a compatible OpenRouter model —";
export const NO_MODEL = "— no compatible text-output model —";
export const CATALOG_ERROR = "— OpenRouter model catalog unavailable —";
export const MEDIA_SPECS = [
    { modality: "image", type: "IMAGE", names: ["image", "image_2", "image_3"] },
    { modality: "video", type: "VIDEO", names: ["video", "video_2", "video_3"] },
    { modality: "audio", type: "AUDIO", names: ["audio", "audio_2", "audio_3"] },
];
export const MEDIA_INPUTS = new Set(MEDIA_SPECS.flatMap((spec) => spec.names));
const SEEDLESS_WIDGET_COUNT = 9;
const CURRENT_WIDGET_COUNT = 10;
const SEED_WIDGET_INDEX = 2;
const REGENERATE_WIDGET_INDEX = 7;

export function migrateLegacyWidgetValues(serializedNode) {
    const values = serializedNode?.widgets_values;
    if (!Array.isArray(values)) return;

    const isSeedEra = values.length === CURRENT_WIDGET_COUNT
        && typeof values[5] === "number"
        && typeof values[6] === "string"
        && typeof values[7] === "boolean";
    if (isSeedEra) values.splice(SEED_WIDGET_INDEX, 1);

    const isSeedless = values.length === SEEDLESS_WIDGET_COUNT
        && typeof values[5] === "string"
        && typeof values[6] === "boolean"
        && typeof values[7] === "string"
        && typeof values[8] === "string";
    if (isSeedless) values.splice(REGENERATE_WIDGET_INDEX, 0, true);
}

export function mediaModality(name) {
    return MEDIA_SPECS.find((spec) => spec.names.includes(name))?.modality ?? null;
}

export function requiredModalities(inputs = []) {
    const required = new Set(["text"]);
    for (const input of inputs) {
        const modality = mediaModality(input?.name);
        if (modality && input.link != null) {
            required.add(modality);
        }
    }
    return required;
}

export function desiredMediaInputNames(inputs = []) {
    const linksByName = new Map(inputs.map((input) => [input?.name, input?.link]));
    const desired = new Set();
    for (const spec of MEDIA_SPECS) {
        let highestConnected = -1;
        spec.names.forEach((name, index) => {
            if (linksByName.get(name) != null) highestConnected = index;
        });
        const visibleCount = Math.min(spec.names.length, Math.max(1, highestConnected + 2));
        spec.names.slice(0, visibleCount).forEach((name) => desired.add(name));
    }
    return desired;
}

export function isModelFree(model) {
    if (!model) return false;
    if (typeof model === "string") return model.includes(":free");
    return Boolean(model.is_free) || String(model.id || "").includes(":free");
}

export function compatibleModels(models, required, freeOnly = false) {
    return models
        .filter((model) => {
            const inputs = new Set(model.input_modalities || []);
            const outputs = new Set(model.output_modalities || []);
            const matches = outputs.has("text") && [...required].every((item) => inputs.has(item));
            if (!matches) return false;
            if (freeOnly && !isModelFree(model)) return false;
            return true;
        })
        .sort((left, right) => {
            const leftFree = isModelFree(left) ? 0 : 1;
            const rightFree = isModelFree(right) ? 0 : 1;
            if (leftFree !== rightFree) {
                return leftFree - rightFree;
            }
            return (left.id || "").localeCompare(right.id || "");
        })
        .map((model) => model.id);
}

export function nextModelValue(current, compatible) {
    return compatible.includes(current) ? current : CHOOSE_MODEL;
}

export function ensureApiKeyWidget(node) {
    if (!node) return null;
    let widget = node.widgets?.find((w) => w.name === "api_key");
    if (!widget) {
        const inputIndex = node.inputs?.findIndex((input) => input?.name === "api_key");
        if (inputIndex !== -1 && inputIndex !== undefined) {
            const input = node.inputs[inputIndex];
            if (input?.link != null) return null;
            if (typeof node.removeInput === "function") {
                node.removeInput(inputIndex);
            }
        }
        if (typeof node.addWidget === "function") {
            widget = node.addWidget("text", "api_key", "", () => {}, {
                multiline: false,
                placeholder: "OpenRouter API Key (sk-or-v1-...)",
                tooltip: "Enter your OpenRouter API key here or leave blank to use OPENROUTER_API_KEY environment variable.",
            });
        }
    }
    return widget;
}
