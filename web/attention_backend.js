import { app } from "../../../scripts/app.js";

const SOL_WIDGETS = ["sol_tau", "sol_start_percent", "sol_end_percent"];

function setWidgetHidden(widget, hidden) {
    if (!widget) return;
    if (hidden) {
        if (!widget.__h3V100Original) {
            widget.__h3V100Original = {
                type: widget.type,
                computeSize: widget.computeSize,
            };
        }
        widget.type = "hidden";
        widget.hidden = true;
        widget.computeSize = () => [0, -4];
    } else if (widget.__h3V100Original) {
        widget.type = widget.__h3V100Original.type;
        widget.hidden = false;
        widget.computeSize = widget.__h3V100Original.computeSize;
    }
}

app.registerExtension({
    name: "H3.V100.AttentionControls",
    nodeCreated(node) {
        if (node.comfyClass !== "H3V100Optimize") return;
        const mode = node.widgets?.find(
            (widget) => widget.name === "attention_backend",
        );
        if (!mode) return;

        const update = () => {
            const showSol = mode.value === "sol_attn";
            for (const name of SOL_WIDGETS) {
                setWidgetHidden(
                    node.widgets?.find((widget) => widget.name === name),
                    !showSol,
                );
            }
            const computed = node.computeSize();
            node.setSize([Math.max(node.size[0], computed[0]), computed[1]]);
            node.graph?.setDirtyCanvas(true, true);
        };

        const originalCallback = mode.callback;
        mode.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            update();
            return result;
        };
        setTimeout(update, 0);
        setTimeout(update, 100);
    },
});
