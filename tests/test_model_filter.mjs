import assert from "node:assert/strict";
import test from "node:test";

import {
    CATALOG_ERROR,
    CHOOSE_MODEL,
    compatibleModels,
    desiredMediaInputNames,
    ensureApiKeyWidget,
    isModelFree,
    migrateLegacyWidgetValues,
    nextModelValue,
    requiredModalities,
} from "../web/model_filter.mjs";

test("seed-era saved nodes remove seed and default regenerate on", () => {
    const seedEra = {
        widgets_values: [
            "google/model",
            "auto",
            471994551739533,
            120,
            1.0,
            4096,
            "text",
            false,
            "system",
            "user",
        ],
    };
    migrateLegacyWidgetValues(seedEra);
    assert.deepEqual(seedEra.widgets_values, [
        "google/model",
        "auto",
        120,
        1.0,
        4096,
        "text",
        false,
        true,
        "system",
        "user",
    ]);
});

test("seedless saved nodes gain regenerate without shifting prompts", () => {
    const seedless = {
        widgets_values: [
            "google/model", "auto", 120, 1.0, 4096, "text", false, "system", "user",
        ],
    };
    migrateLegacyWidgetValues(seedless);
    assert.deepEqual(seedless.widgets_values, [
        "google/model", "auto", 120, 1.0, 4096, "text", false, true, "system", "user",
    ]);
});

test("current saved nodes retain an explicit regenerate value", () => {
    const current = {
        widgets_values: [
            "google/model", "auto", 120, 1.0, 4096, "text", false, false, "system", "user",
        ],
    };
    migrateLegacyWidgetValues(current);
    assert.deepEqual(current.widgets_values, [
        "google/model", "auto", 120, 1.0, 4096, "text", false, false, "system", "user",
    ]);
});

const models = [
    { id: "text", input_modalities: ["text"], output_modalities: ["text"] },
    { id: "vision", input_modalities: ["text", "image"], output_modalities: ["text"] },
    { id: "video", input_modalities: ["text", "image", "video"], output_modalities: ["text"] },
    { id: "omni", input_modalities: ["text", "image", "video", "audio"], output_modalities: ["text"] },
    { id: "generator", input_modalities: ["text", "image", "video", "audio"], output_modalities: ["image"] },
];

test("connected modalities are intersected", () => {
    const required = requiredModalities([
        { name: "image_2", link: 2 },
        { name: "video", link: 3 },
        { name: "audio_3", link: null },
    ]);
    assert.deepEqual([...required].sort(), ["image", "text", "video"]);
    assert.deepEqual(compatibleModels(models, required), ["omni", "video"]);
});

test("progressive sockets retain one empty successor per modality", () => {
    const initial = desiredMediaInputNames([
        { name: "image", link: null },
        { name: "image_2", link: null },
        { name: "image_3", link: null },
        { name: "video", link: null },
        { name: "video_2", link: null },
        { name: "video_3", link: null },
        { name: "audio", link: null },
        { name: "audio_2", link: null },
        { name: "audio_3", link: null },
    ]);
    assert.deepEqual([...initial], ["image", "video", "audio"]);

    const growing = desiredMediaInputNames([
        { name: "image", link: 1 },
        { name: "image_2", link: 2 },
        { name: "video", link: 3 },
        { name: "audio", link: null },
    ]);
    assert.deepEqual(
        [...growing],
        ["image", "image_2", "image_3", "video", "video_2", "audio"],
    );
});

test("a restored non-contiguous link never loses its socket", () => {
    const desired = desiredMediaInputNames([
        { name: "image", link: null },
        { name: "image_2", link: 9 },
        { name: "video", link: null },
        { name: "audio", link: null },
    ]);
    assert.deepEqual([...desired], ["image", "image_2", "image_3", "video", "audio"]);
});

test("all three media inputs require an omni input model", () => {
    const required = new Set(["text", "image", "video", "audio"]);
    assert.deepEqual(compatibleModels(models, required), ["omni"]);
});

test("an invalid paid model is never silently replaced", () => {
    assert.equal(nextModelValue("vision", ["omni"]), CHOOSE_MODEL);
    assert.equal(nextModelValue("omni", ["omni"]), "omni");
});

test("catalog errors are distinct from an empty compatibility result", () => {
    assert.notEqual(CATALOG_ERROR, "— no compatible text-output model —");
});


test("key-era saved nodes preserve every widget including the appended API key", () => {
    const values = ["vendor/model", "auto", 120, 1, 4096, "text", false, true, "system", "user", "test-key"];
    const node = { widgets_values: [...values] };
    migrateLegacyWidgetValues(node);
    migrateLegacyWidgetValues(node);
    assert.deepEqual(node.widgets_values, values);
});

test("ensureApiKeyWidget creates api_key widget on node if not present", () => {
    const widgets = [];
    const node = {
        widgets,
        inputs: [],
        addWidget(type, name, value, cb, opts) {
            const w = { type, name, value, cb, opts };
            widgets.push(w);
            return w;
        },
    };
    const widget = ensureApiKeyWidget(node);
    assert.equal(widget.name, "api_key");
    assert.equal(widgets.length, 1);
    const second = ensureApiKeyWidget(node);
    assert.equal(second, widget);
    assert.equal(widgets.length, 1);
});

test("ensureApiKeyWidget converts unconnected api_key input slot to widget", () => {
    let removed = false;
    const widgets = [];
    const inputs = [{ name: "api_key", link: null }];
    const node = {
        widgets,
        inputs,
        removeInput(idx) {
            inputs.splice(idx, 1);
            removed = true;
        },
        addWidget(type, name, value, cb, opts) {
            const w = { type, name, value, cb, opts };
            widgets.push(w);
            return w;
        },
    };
    ensureApiKeyWidget(node);
    assert.equal(removed, true);
    assert.equal(inputs.length, 0);
    assert.equal(widgets.length, 1);
    assert.equal(widgets[0].name, "api_key");
});

test("ensureApiKeyWidget preserves connected api_key input slot", () => {
    const widgets = [];
    const inputs = [{ name: "api_key", link: 42 }];
    const node = {
        widgets,
        inputs,
        removeInput() {
            assert.fail("should not remove connected input");
        },
        addWidget() {
            assert.fail("should not add widget when input is wired");
        },
    };
    const result = ensureApiKeyWidget(node);
    assert.equal(result, null);
    assert.equal(inputs.length, 1);
    assert.equal(widgets.length, 0);
});

test("isModelFree detects free models by id or property", () => {
    assert.equal(isModelFree({ id: "meta-llama/llama-3.3-70b-instruct:free", is_free: true }), true);
    assert.equal(isModelFree({ id: "google/gemini-2.0-flash-exp:free" }), true);
    assert.equal(isModelFree({ id: "anthropic/claude-3.5-sonnet", is_free: false }), false);
    assert.equal(isModelFree("meta-llama/llama-3.3-70b-instruct:free"), true);
    assert.equal(isModelFree("openai/gpt-4o"), false);
});

test("compatibleModels sorts free models first and filters by freeOnly", () => {
    const mixed = [
        { id: "paid/z-model", input_modalities: ["text"], output_modalities: ["text"], is_free: false },
        { id: "free/b-model:free", input_modalities: ["text"], output_modalities: ["text"], is_free: true },
        { id: "paid/a-model", input_modalities: ["text"], output_modalities: ["text"], is_free: false },
        { id: "free/a-model:free", input_modalities: ["text"], output_modalities: ["text"], is_free: true },
    ];
    const required = new Set(["text"]);
    // Free models come first sorted alphabetically, followed by paid models sorted alphabetically
    const all = compatibleModels(mixed, required, false);
    assert.deepEqual(all, ["free/a-model:free", "free/b-model:free", "paid/a-model", "paid/z-model"]);

    // With freeOnly = true, paid models are omitted
    const freeOnly = compatibleModels(mixed, required, true);
    assert.deepEqual(freeOnly, ["free/a-model:free", "free/b-model:free"]);
});

