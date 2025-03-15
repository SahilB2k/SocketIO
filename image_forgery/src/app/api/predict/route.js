import { NextResponse } from "next/server";
import tf from "@tensorflow/tfjs-node";

export async function POST(req) {
    const formData = await req.formData();
    const file = formData.get("image");
    
    if (!file) {
        return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    const model = await tf.loadLayersModel("file://./model/model.json"); // Adjust path based on model location
    const buffer = Buffer.from(await file.arrayBuffer());
    const tensor = tf.node.decodeImage(buffer).resizeNearestNeighbor([224, 224]).expandDims();
    const prediction = model.predict(tensor).dataSync();

    return NextResponse.json({ result: prediction[0] > 0.5 ? "Fake" : "Real" });
}
