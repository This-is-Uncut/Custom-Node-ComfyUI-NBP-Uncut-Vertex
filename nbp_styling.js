import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Uncut.NBP.Styling",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NanoBananaProNode" || nodeData.name === "NanoBananaProNodeVertex") {
            // Sets the default background color to R40 G120 B120 (#287878)
            nodeType.prototype.color = "#287878";
            nodeType.prototype.bgcolor = "#287878";
        }
    }
});