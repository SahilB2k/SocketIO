"use client"; // If using App Router

import { useState } from "react";

export default function UploadPage() {
    const [selectedImage, setSelectedImage] = useState(null);
    const [result, setResult] = useState("");

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        setSelectedImage(file);
    };

    const handleUpload = async () => {
        if (!selectedImage) return;

        const formData = new FormData();
        formData.append("image", selectedImage);

        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        setResult(data.result);
    };

    return (
        <div className="flex flex-col items-center justify-center h-screen">
            <h1 className="text-2xl font-bold">Image Forgery Detection</h1>

            <input type="file" onChange={handleImageChange} className="my-4" />
            {selectedImage && <img src={URL.createObjectURL(selectedImage)} alt="Preview" className="w-48 h-48 object-cover rounded-md" />}

            <button onClick={handleUpload} className="bg-blue-500 text-white px-4 py-2 mt-4 rounded">
                Upload & Detect
            </button>

            {result && <p className="mt-4 font-semibold">Result: {result}</p>}
        </div>
    );
}
